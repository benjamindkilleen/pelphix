from typing import Optional, Type, List
import logging
from pathlib import Path
from deepdrr import geo
from deepdrr.utils import data_utils
from stringcase import snakecase, camelcase

from .tool import Tool
from ..utils import get_drawings_dir

log = logging.getLogger(__name__)


class Screw(Tool):
    pass

    # Populate these fields in the subclass
    url: str

    def download(self):
        """Download the model files using the OneDrive API.

        Note: You must have the rights to these model files.
        """
        filename = f"steel.stl"
        # TODO: handle no permissions
        data_utils.download(
            f"{self.url}&download=1",
            filename,
            root=get_drawings_dir() / snakecase(camelcase(self.__class__.__name__)),
        )

    def __init__(self, *args, **kwargs):
        self.download()
        super().__init__(*args, **kwargs)


class OrthopedicScrew(Tool):
    """An orthopedic screw.

    Based on marinor0

    Obtain the rights to these model files here: https://www.cgtrader.com/3d-print-models/science/biology/orthopedic-screw

    Once you have the rights to the model files, email proof of purchase to killeen@jhu.edu to get an updated link.

    """

    radius = 2.5  # not really important
    tip = geo.point(4, 0, 4)

    @classmethod
    def length_mm(cls) -> int:
        classname = cls.__name__
        return int(classname[5:])

    @property
    def base(self) -> geo.Point3D:
        return geo.point(4, self.length_mm(), 4)


class Screw60(OrthopedicScrew):
    # Valid until Sep 1, 2023
    url = "https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/bkillee1_jh_edu/ESFD4DzmYzBEhEQLzFWOPIcBmaD6KVir3fxSEcfTRSUA3g?e=ZgUl2f"


class Screw65(OrthopedicScrew):
    # Valid until Sep 1, 2023
    url = "https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/bkillee1_jh_edu/EZep0p3xEDZAhd8uzkVqDP0BJkOPnbIgaP0DTON4qGxVZw?e=s5lcVe"


class Screw70(OrthopedicScrew):
    # Valid until Sep 1, 2023
    url = "https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/bkillee1_jh_edu/EX040XMNtT1Hq4dAey6EXToBhd58z2LOUrnDzdbtSb5NyQ?e=ct3KFA"


class Screw75(OrthopedicScrew):
    # Valid until Sep 1, 2023
    url = "https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/bkillee1_jh_edu/EUfXi_0WGGtPhfBcppp9SZYBDZagFJh0f8dGHiz9phsE6g?e=o3qdOh"


class Screw80(OrthopedicScrew):
    # Valid until Sep 1, 2023
    url = "https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/bkillee1_jh_edu/ETppCsaVRMpMhuBOS9nV0vABiv0jkl3RLo9bvod9jQ2_Aw?e=k4m30J"


class CannulatedScrew(Screw):
    radius = 3.25

    @classmethod
    def length_mm(cls) -> int:
        classname = cls.__name__
        return int(classname.split("_")[2][1:])

    @property
    def tip(self) -> geo.Point3D:
        return geo.point(4.115, self.length_mm(), 4.115)

    base = geo.point(4.115, 3, 4.115)


class Screw_T16_L30(CannulatedScrew):
    url = "https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/bkillee1_jh_edu/ERcgOZdKlKlEqtN5VVKfSc4Bh5jxZxz3lIRogjWuzWUjTQ?e=WMZxVA"


class Screw_T16_L40(CannulatedScrew):
    url = "https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/bkillee1_jh_edu/ERyj_Ik8f45JiJgBYjh6aiYBpf2NZE2mxaVKWOHMNVihhQ?e=QC3iBn"


class Screw_T16_L50(CannulatedScrew):
    url = "https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/bkillee1_jh_edu/EeZjWPwJxhdMipzdBlKzmPEBXv9ftHA_jIedhVq-1gJUJQ?e=82qErf"


class Screw_T16_L60(CannulatedScrew):
    url = "https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/bkillee1_jh_edu/Ec6cMtGJj8FFpbkU-SpwTnMBunI6ndBlPUokHpbTsnc94g?e=ABMKQW"


class Screw_T16_L70(CannulatedScrew):
    url = "https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/bkillee1_jh_edu/ESEps4eQ6IFMsVsdLQNGolMB4lg8f6m8o6ySMylFqTwp5Q?e=IaPfjK"


class Screw_T16_L80(CannulatedScrew):
    url = "https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/bkillee1_jh_edu/EUYiI7OcnlVAvGJnpkgiIaABpEDebXap_e9HOzUpznJmIg?e=MHLGfb"


class Screw_T16_L90(CannulatedScrew):
    url = "https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/bkillee1_jh_edu/EZhsxoareOdJvYTFU-TVelsBA_wzCpGuaNAG8xA3ONrKXA?e=at0nqf"


class Screw_T16_L100(CannulatedScrew):
    url = "https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/bkillee1_jh_edu/EY9iOh0hxAZLosgo1eEFn6oB6DqLXjafgWcXzY-tEHl0mw?e=MYUVjV"


class Screw_T16_L110(CannulatedScrew):
    url = "https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/bkillee1_jh_edu/EdTzuYd6k6BGrEpobQ8nZn4BcynxmbemLMYoxzkYh5u73Q?e=CQBozs"


class Screw_T16_L120(CannulatedScrew):
    url = "https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/bkillee1_jh_edu/EYr95BywkstAqdntdw7i2n0BpPLRoQcJfqwz7x2i2pa6Rw?e=yTRjDO"


class Screw_T16_L130(CannulatedScrew):
    url = "https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/bkillee1_jh_edu/EXvbZJUATINKiXbTk0FtwiIBrHJEWpMxhzRnl2RAFWbQmg?e=xtcuQb"


def get_screw(length: int) -> Type[CannulatedScrew]:
    """Get the longest cannulated screw type as long as but not exceeding `length`."""
    if length <= 30:
        log.warning("Corridor length is less than 30mm. Using 30mm screw.")
        return Screw_T16_L30
    elif length <= 40:
        return Screw_T16_L30
    elif length <= 50:
        return Screw_T16_L40
    elif length <= 60:
        return Screw_T16_L50
    elif length <= 70:
        return Screw_T16_L60
    elif length <= 80:
        return Screw_T16_L70
    elif length <= 90:
        return Screw_T16_L80
    elif length <= 100:
        return Screw_T16_L90
    elif length <= 110:
        return Screw_T16_L100
    elif length <= 120:
        return Screw_T16_L110
    elif length <= 130:
        return Screw_T16_L120
    else:
        return Screw_T16_L130


def get_screw_choices(length: Optional[int] = None) -> set[Type[CannulatedScrew]]:
    """Get a list of all the screws shorter than `length`."""
    screws: set[Type[CannulatedScrew]] = {
        Screw_T16_L30,
        Screw_T16_L40,
        Screw_T16_L50,
        Screw_T16_L60,
        Screw_T16_L70,
        Screw_T16_L80,
        Screw_T16_L90,
        Screw_T16_L100,
        Screw_T16_L110,
        Screw_T16_L120,
        Screw_T16_L130,
    }
    return set(screw for screw in screws if length is None or screw.length_mm() <= length)
