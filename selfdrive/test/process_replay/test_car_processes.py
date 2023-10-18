import unittest
import pytest
import sys

from parameterized import parameterized_class
from typing import List, Optional

from openpilot.selfdrive.car.car_helpers import interface_names
from openpilot.selfdrive.test.process_replay.process_replay import check_openpilot_enabled
from openpilot.selfdrive.test.process_replay.helpers import TestProcessReplayDiffBase


source_segments = [
  ("BODY", "937ccb7243511b65|2022-05-24--16-03-09--1"),        # COMMA.BODY
  ("HYUNDAI", "02c45f73a2e5c6e9|2021-01-01--19-08-22--1"),     # HYUNDAI.SONATA
  ("HYUNDAI2", "d545129f3ca90f28|2022-11-07--20-43-08--3"),    # HYUNDAI.KIA_EV6 (+ QCOM GPS)
  ("TOYOTA", "0982d79ebb0de295|2021-01-04--17-13-21--13"),     # TOYOTA.PRIUS
  ("TOYOTA2", "0982d79ebb0de295|2021-01-03--20-03-36--6"),     # TOYOTA.RAV4
  ("TOYOTA3", "f7d7e3538cda1a2a|2021-08-16--08-55-34--6"),     # TOYOTA.COROLLA_TSS2
  ("HONDA", "eb140f119469d9ab|2021-06-12--10-46-24--27"),      # HONDA.CIVIC (NIDEC)
  ("HONDA2", "7d2244f34d1bbcda|2021-06-25--12-25-37--26"),     # HONDA.ACCORD (BOSCH)
  ("CHRYSLER", "4deb27de11bee626|2021-02-20--11-28-55--8"),    # CHRYSLER.PACIFICA_2018_HYBRID
  ("RAM", "17fc16d840fe9d21|2023-04-26--13-28-44--5"),         # CHRYSLER.RAM_1500
  ("SUBARU", "341dccd5359e3c97|2022-09-12--10-35-33--3"),      # SUBARU.OUTBACK
  ("GM", "0c58b6a25109da2b|2021-02-23--16-35-50--11"),         # GM.VOLT
  ("GM2", "376bf99325883932|2022-10-27--13-41-22--1"),         # GM.BOLT_EUV
  ("NISSAN", "35336926920f3571|2021-02-12--18-38-48--46"),     # NISSAN.XTRAIL
  ("VOLKSWAGEN", "de9592456ad7d144|2021-06-29--11-00-15--6"),  # VOLKSWAGEN.GOLF
  ("MAZDA", "bd6a637565e91581|2021-10-30--15-14-53--4"),       # MAZDA.CX9_2021
  ("FORD", "54827bf84c38b14f|2023-01-26--21-59-07--4"),        # FORD.BRONCO_SPORT_MK1

  # Enable when port is tested and dashcamOnly is no longer set
  #("TESLA", "bb50caf5f0945ab1|2021-06-19--17-20-18--3"),      # TESLA.AP2_MODELS
  #("VOLKSWAGEN2", "3cfdec54aa035f3f|2022-07-19--23-45-10--2"),  # VOLKSWAGEN.PASSAT_NMS
]

segments = [
  ("BODY", "aregenECF15D9E559|2023-05-10--14-26-40--0"),
  ("HYUNDAI", "aregenAB9F543F70A|2023-05-10--14-28-25--0"),
  ("HYUNDAI2", "aregen39F5A028F96|2023-05-10--14-31-00--0"),
  ("TOYOTA", "aregen8D6A8B36E8D|2023-05-10--14-32-38--0"),
  ("TOYOTA2", "aregenB1933C49809|2023-05-10--14-34-14--0"),
  ("TOYOTA3", "aregen5D9915223DC|2023-05-10--14-36-43--0"),
  ("HONDA", "aregen484B732B675|2023-05-10--14-38-23--0"),
  ("HONDA2", "aregenAF6ACED4713|2023-05-10--14-40-01--0"),
  ("CHRYSLER", "aregen99B094E1E2E|2023-05-10--14-41-40--0"),
  ("RAM", "aregen5C2487E1EEB|2023-05-10--14-44-09--0"),
  ("SUBARU", "aregen98D277B792E|2023-05-10--14-46-46--0"),
  ("GM", "aregen377BA28D848|2023-05-10--14-48-28--0"),
  ("GM2", "aregen7CA0CC0F0C2|2023-05-10--14-51-00--0"),
  ("NISSAN", "aregen7097BF01563|2023-05-10--14-52-43--0"),
  ("VOLKSWAGEN", "aregen765AF3D2CB5|2023-05-10--14-54-23--0"),
  ("MAZDA", "aregen3053762FF2E|2023-05-10--14-56-53--0"),
  ("FORD", "aregenDDE0F89FA1E|2023-05-10--14-59-26--0"),
]

# dashcamOnly makes don't need to be tested until a full port is done
excluded_interfaces = ["mock", "tesla"]

ALL_CARS = sorted({car for car, _ in segments})

CAR_TO_SEGMENT = dict(segments)


@pytest.mark.slow
@parameterized_class(('case_name', 'segment'), [(i, CAR_TO_SEGMENT[i]) for i in ALL_CARS])
class TestCarProcessReplay(TestProcessReplayDiffBase):
  """
  Runs a replay diff on a segment for each car.
  """

  case_name: Optional[str] = None
  tested_cars: List[str] = ALL_CARS

  @classmethod
  def setUpClass(cls):
    if cls.case_name not in cls.tested_cars:
      raise unittest.SkipTest(f"{cls.case_name} was not requested to be tested")
    super().setUpClass()

  def test_all_makes_are_tested(self):
    if self.tested_cars != ALL_CARS:
      raise unittest.SkipTest("skipping check because some cars were skipped via command line")

    # check to make sure all car brands are tested
    untested = (set(interface_names) - set(excluded_interfaces)) - {c.lower() for c in self.tested_cars}
    self.assertEqual(len(untested), 0, f"Cars missing routes: {str(untested)}")

  def test_controlsd_engaged(self):
    if "controlsd" not in self.tested_procs:
      raise unittest.SkipTest("controlsd was not requested to be tested")

    # check to make sure openpilot is engaged in the route
    log_msgs = self.log_msgs["controlsd"]
    self.assertTrue(check_openpilot_enabled(log_msgs), f"Route did not enable at all or for long enough: {self.segment}")


if __name__ == '__main__':
  pytest.main([*sys.argv[1:], __file__])