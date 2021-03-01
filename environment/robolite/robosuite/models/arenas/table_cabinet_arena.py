import numpy as np
from robosuite.models.arenas import Arena
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.utils.mjcf_utils import array_to_string, string_to_array


class TableCabinetArena(Arena):
    """Workspace that contains an empty table."""

    def __init__(
        self, table_full_size=(0.8, 0.8, 0.8), table_friction=(1, 0.005, 0.0001)
    ):
        """
        Args:
            table_full_size: full dimensions of the table
            friction: friction parameters of the table
        """
        super().__init__(xml_path_completion("arenas/table_cabinet_arena.xml"))

        self.table_full_size = np.array(table_full_size)
        self.table_half_size = self.table_full_size / 2
        self.table_friction = table_friction

        self.floor = self.worldbody.find("./geom[@name='floor']")
        self.table_body = self.worldbody.find("./body[@name='table']")
        self.table_collision = self.table_body.find("./geom[@name='table_collision']")
        self.table_visual = self.table_body.find("./geom[@name='table_visual']")
        self.table_top = self.table_body.find("./site[@name='table_top']")
        self.door_body = self.worldbody.find("./body[@name='frame_link']")
        self.door_inertial = self.door_body.find("./inertial")  # since the mass cannot be directly set to body, and the inertial in xml does not have a name, the inertial needs to be referred here
        self.door_link = self.door_body.find(("./body[@name='door_link']"))
        self.door_hinge = self.door_link.find(("./joint[@name='hinge0']"))
        # self.door_hinge = self.door_body.find("./body[@name='door_link2']").find(("./joint[@name='base_to_door2']"))
        self.knob_link_body = self.door_link.find("./body[@name='knob_link']")
        self.knob_link_inertial = self.knob_link_body.find("./inertial")
        self.knob_geom = self.knob_link_body.find("./geom[@name='cabinet_knob']")
        assert self.floor is not None
        assert self.table_body is not None
        assert self.door_body is not None
        assert self.door_inertial is not None
        assert self.door_link is not None
        assert self.door_hinge is not None
        assert self.knob_link_body is not None
        assert self.knob_link_inertial is not None
        assert self.knob_geom is not None

        self.configure_location()

    def configure_location(self):
        self.bottom_pos = np.array([0, 0, 0])
        self.floor.set("pos", array_to_string(self.bottom_pos))

        self.center_pos = self.bottom_pos + np.array([0, 0, self.table_half_size[2]])
        self.table_body.set("pos", array_to_string(self.center_pos))
        self.table_collision.set("size", array_to_string(self.table_half_size))
        self.table_collision.set("friction", array_to_string(self.table_friction))
        self.table_visual.set("size", array_to_string(self.table_half_size))

        self.table_top.set(
            "pos", array_to_string(np.array([0, 0, self.table_half_size[2]]))
        )

    @property
    def table_top_abs(self):
        """Returns the absolute position of table top"""
        table_height = np.array([0, 0, self.table_full_size[2]])
        return string_to_array(self.floor.get("pos")) + table_height
