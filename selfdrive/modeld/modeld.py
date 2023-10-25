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
import logging
logging.basicConfig(filename='/tmp/myapp.log', level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger=logging.getLogger(__name__)

SEND_RAW_PRED = True #os.getenv('SEND_RAW_PRED')

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
  inputs: Dict[str, np.ndarray]
  output: np.ndarray
  prev_desire: np.ndarray  # for tracking the rising edge of the pulse
  model: ModelRunner

  def __init__(self, context: CLContext):
    self.cnt = 0
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


  raw_preds = []
  raw_preds_prev = []
  total_cnt = 0
  total_err_cnt = 0
  cl_context = CLContext()
  model = ModelState(cl_context)
  cloudlog.warning("models loaded, modeld starting")

  while True:
    # TODO: path planner timeout?
    
    mt1 = time.perf_counter()
    model_output = model.run()
    raw_preds.append(np.copy(model_output['raw_pred']))
    if len(raw_preds) == 10:
      if len(raw_preds_prev) == len(raw_preds):
        for i in range(len(raw_preds)):
          try:
            assert len(raw_preds[i]) > 0
            a = raw_preds[i]
            b = raw_preds_prev[i]
            equal = a == b
            assert np.all(equal)
            assert max(a-b) == 0
          except Exception as e:
            unequal_idxs = np.where(0 == equal)[0]
            cloudlog.error(f'ERROR: {e}')
            cloudlog.error(f'UNEQUAL IDXS: {unequal_idxs}')
            logger.error(f'ERROR: {e}')
            logger.error(f'UNEQUAL IDXS: {unequal_idxs}')
            total_err_cnt += 1
      raw_preds_prev = raw_preds
      raw_preds = []
      total_cnt += 1
      cloudlog.warning(f'DID {total_cnt} ITERATIONS with {total_err_cnt} errors')
      logger.error(f'DID {total_cnt} ITERATIONS with {total_err_cnt} errors')
      #ModelFrame(cl_context)

    mt2 = time.perf_counter()
    model_execution_time = mt2 - mt1

if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    sys.exit()
