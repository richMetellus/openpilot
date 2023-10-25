#!/usr/bin/env python3
import os
import os
os.environ['SEND_RAW_PRED'] = '1'

import sys
import time
from collections import defaultdict
from typing import Any
from itertools import zip_longest
from tqdm import tqdm
import numpy as np

import cereal.messaging as messaging
from cereal.visionipc import VisionIpcServer, VisionStreamType
from common.spinner import Spinner
from common.timeout import Timeout
from common.transformations.camera import tici_f_frame_size, tici_d_frame_size
from system.hardware import PC
from selfdrive.manager.process_config import managed_processes
from selfdrive.test.openpilotci import BASE_URL, get_url
from selfdrive.test.process_replay.test_processes import format_diff
from system.version import get_commit
from tools.lib.framereader import FrameReader
from tools.lib.logreader import LogReader

TEST_ROUTE = "4cf7a6ad03080c90|2021-09-29--13-46-36"
SEGMENT = 0
MAX_FRAMES = 600

VIPC_STREAM = {"roadCameraState": VisionStreamType.VISION_STREAM_ROAD, "driverCameraState": VisionStreamType.VISION_STREAM_DRIVER,
               "wideRoadCameraState": VisionStreamType.VISION_STREAM_WIDE_ROAD}
def get_log_fn(ref_commit, test_route):
  return f"{test_route}_model_tici_{ref_commit}.bz2"


def replace_calib(msg, calib):
  msg = msg.as_builder()
  if calib is not None:
    msg.liveCalibration.rpyCalib = calib.tolist()
  return msg


def model_replay(lr):
  if not PC:
    spinner = Spinner()
    spinner.update("starting model replay")
  else:
    spinner = None

  vipc_server = VisionIpcServer("camerad")
  vipc_server.create_buffers(VisionStreamType.VISION_STREAM_ROAD, 40, False, *(tici_f_frame_size))
  vipc_server.create_buffers(VisionStreamType.VISION_STREAM_DRIVER, 40, False, *(tici_d_frame_size))
  vipc_server.create_buffers(VisionStreamType.VISION_STREAM_WIDE_ROAD, 40, False, *(tici_f_frame_size))
  vipc_server.start_listener()

  sm = messaging.SubMaster(['modelV2', 'driverStateV2'])
  pm = messaging.PubMaster(['roadCameraState', 'wideRoadCameraState', 'driverCameraState', 'liveCalibration', 'lateralPlan'])

  while True:
    managed_processes['modeld'].start()
    time.sleep(10)
    managed_processes['modeld'].stop()


  return log_msgs


if __name__ == "__main__":

  update = "--update" in sys.argv
  replay_dir = os.path.dirname(os.path.abspath(__file__))
  ref_commit_fn = os.path.join(replay_dir, "model_replay_ref_commit")

  # load logs
  lr = list(LogReader(get_url(TEST_ROUTE, SEGMENT)))

  # run replay
  while True:
    log_msgs = model_replay(lr)
