#!/usr/bin/env python3
import sys
import os
import time
import pickle
import numpy as np
import cereal.messaging as messaging
from pathlib import Path
from typing import Dict, Optional
from setproctitle import setproctitle
from cereal.messaging import PubMaster, SubMaster
from cereal.visionipc import VisionIpcClient, VisionStreamType, VisionBuf
from openpilot.system.swaglog import cloudlog
from openpilot.common.params import Params
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import config_realtime_process
from openpilot.common.transformations.model import get_warp_matrix
from openpilot.selfdrive.modeld.runners import ModelRunner, Runtime
from openpilot.selfdrive.modeld.parse_model_outputs import Parser
from openpilot.selfdrive.modeld.fill_model_msg import fill_model_msg, fill_pose_msg, PublishState
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.modeld.models.commonmodel_pyx import ModelFrame, CLContext

SEND_RAW_PRED = os.getenv('SEND_RAW_PRED')

MODEL_PATHS = {
  ModelRunner.THNEED: Path(__file__).parent / 'models/supercombo.thneed',
  ModelRunner.ONNX: Path(__file__).parent / 'models/supercombo.onnx'}

METADATA_PATH = Path(__file__).parent / 'models/supercombo_metadata.pkl'

class FrameMeta:
  frame_id: int = 0
  timestamp_sof: int = 0
  timestamp_eof: int = 0

  def __init__(self, vipc=None):
    if vipc is not None:
      self.frame_id, self.timestamp_sof, self.timestamp_eof = vipc.frame_id, vipc.timestamp_sof, vipc.timestamp_eof

class ModelState:
  frame: ModelFrame
  wide_frame: ModelFrame
  inputs: Dict[str, np.ndarray]
  output: np.ndarray
  prev_desire: np.ndarray  # for tracking the rising edge of the pulse
  model: ModelRunner

  def __init__(self, context: CLContext):
    self.cnt = 0
    self.frame = ModelFrame(context)
    self.wide_frame = ModelFrame(context)
    self.prev_desire = np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32)
    self.inputs = {
      'desire': np.zeros(ModelConstants.DESIRE_LEN * (ModelConstants.HISTORY_BUFFER_LEN+1), dtype=np.float32),
      'traffic_convention': np.zeros(ModelConstants.TRAFFIC_CONVENTION_LEN, dtype=np.float32),
      'nav_features': np.zeros(ModelConstants.NAV_FEATURE_LEN, dtype=np.float32),
      'nav_instructions': np.zeros(ModelConstants.NAV_INSTRUCTION_LEN, dtype=np.float32),
      'features_buffer': np.zeros(ModelConstants.HISTORY_BUFFER_LEN * ModelConstants.FEATURE_LEN, dtype=np.float32),
    }

    with open(METADATA_PATH, 'rb') as f:
      model_metadata = pickle.load(f)

    self.output_slices = model_metadata['output_slices']
    net_output_size = model_metadata['output_shapes']['outputs'][1]
    self.output = np.zeros(net_output_size, dtype=np.float32)
    self.parser = Parser()

    self.model = ModelRunner(MODEL_PATHS, self.output, Runtime.GPU, False, context)
    self.model.addInput("input_imgs", None)
    self.model.addInput("big_input_imgs", None)
    for k,v in self.inputs.items():
      self.model.addInput(k, v)

  def slice_outputs(self, model_outputs: np.ndarray) -> Dict[str, np.ndarray]:
    parsed_model_outputs = {k: model_outputs[np.newaxis, v] for k,v in self.output_slices.items()}
    if SEND_RAW_PRED:
      parsed_model_outputs['raw_pred'] = model_outputs.copy()
    return parsed_model_outputs

  def run(self,) -> Optional[Dict[str, np.ndarray]]:

    self.cnt += 1
    if self.cnt %2 == 0:
      img_val = 100.0
    else:
      img_val = 0.0
    self.model.setInputBuffer("input_imgs", img_val * np.ones((128 * 256 * 12), dtype=np.float32))
    self.model.setInputBuffer("big_input_imgs", img_val * np.ones((128 * 256 * 12), dtype=np.float32))


    self.model.execute()
    outputs = self.parser.parse_outputs(self.slice_outputs(self.output))
    return outputs


def main():
  cloudlog.bind(daemon="selfdrive.modeld.modeld")
  setproctitle("selfdrive.modeld.modeld")
  config_realtime_process(7, 54)

  cl_context = CLContext()
  model = ModelState(cl_context)
  cloudlog.warning("models loaded, modeld starting")

  # visionipc clients
  while True:
    available_streams = VisionIpcClient.available_streams("camerad", block=False)
    if available_streams:
      use_extra_client = VisionStreamType.VISION_STREAM_WIDE_ROAD in available_streams and VisionStreamType.VISION_STREAM_ROAD in available_streams
      main_wide_camera = VisionStreamType.VISION_STREAM_ROAD not in available_streams
      break
    time.sleep(.1)

  vipc_client_main_stream = VisionStreamType.VISION_STREAM_WIDE_ROAD if main_wide_camera else VisionStreamType.VISION_STREAM_ROAD
  vipc_client_main = VisionIpcClient("camerad", vipc_client_main_stream, True, cl_context)
  vipc_client_extra = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_WIDE_ROAD, False, cl_context)
  cloudlog.warning(f"vision stream set up, main_wide_camera: {main_wide_camera}, use_extra_client: {use_extra_client}")

  while not vipc_client_main.connect(False):
    time.sleep(0.1)
  while use_extra_client and not vipc_client_extra.connect(False):
    time.sleep(0.1)

  cloudlog.warning(f"connected main cam with buffer size: {vipc_client_main.buffer_len} ({vipc_client_main.width} x {vipc_client_main.height})")
  if use_extra_client:
    cloudlog.warning(f"connected extra cam with buffer size: {vipc_client_extra.buffer_len} ({vipc_client_extra.width} x {vipc_client_extra.height})")

  # messaging
  pm = PubMaster(["modelV2", "cameraOdometry"])
  sm = SubMaster(["lateralPlan", "roadCameraState", "liveCalibration", "driverMonitoringState", "navModel", "navInstruction"])

  publish_state = PublishState()
  frame_id = 0
 
  buf_main, buf_extra = None, None
  meta_main = FrameMeta()
  meta_extra = FrameMeta()

  while True:
    # Keep receiving frames until we are at least 1 frame ahead of previous extra frame
    while meta_main.timestamp_sof < meta_extra.timestamp_sof + 25000000:
      buf_main = vipc_client_main.recv()
      meta_main = FrameMeta(vipc_client_main)
      if buf_main is None:
        break

    if buf_main is None:
      cloudlog.error("vipc_client_main no frame")
      continue

    if use_extra_client:
      # Keep receiving extra frames until frame id matches main camera
      while True:
        buf_extra = vipc_client_extra.recv()
        meta_extra = FrameMeta(vipc_client_extra)
        if buf_extra is None or meta_main.timestamp_sof < meta_extra.timestamp_sof + 25000000:
          break

      if buf_extra is None:
        cloudlog.error("vipc_client_extra no frame")
        continue

      if abs(meta_main.timestamp_sof - meta_extra.timestamp_sof) > 10000000:
        cloudlog.error("frames out of sync! main: {} ({:.5f}), extra: {} ({:.5f})".format(
          meta_main.frame_id, meta_main.timestamp_sof / 1e9,
          meta_extra.frame_id, meta_extra.timestamp_sof / 1e9))

    else:
      # Use single camera
      buf_extra = buf_main
      meta_extra = meta_main

    # TODO: path planner timeout?
    sm.update(0)
    desire = sm["lateralPlan"].desire.raw
    is_rhd = sm["driverMonitoringState"].isRHD
    frame_id = sm["roadCameraState"].frameId

    traffic_convention = np.zeros(2)
    traffic_convention[int(is_rhd)] = 1

    vec_desire = np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32)
    if desire >= 0 and desire < ModelConstants.DESIRE_LEN:
      vec_desire[desire] = 1
    
    mt1 = time.perf_counter()
    model_output = model.run()
    mt2 = time.perf_counter()
    model_execution_time = mt2 - mt1

    if model_output is not None:
      modelv2_send = messaging.new_message('modelV2')
      posenet_send = messaging.new_message('cameraOdometry')
      fill_model_msg(modelv2_send, model_output, publish_state, meta_main.frame_id, meta_extra.frame_id, frame_id, 0.0,
                      meta_main.timestamp_eof, 0.0, model_execution_time, False, True)

      fill_pose_msg(posenet_send, model_output, meta_main.frame_id, 0.0, meta_main.timestamp_eof, True)
      pm.send('modelV2', modelv2_send)
      pm.send('cameraOdometry', posenet_send)


if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    sys.exit()
