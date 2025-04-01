from dataclasses import dataclass


@dataclass
class Offsets:
    """
    Data class for PCB parameters.
    """

    # fmt: off
    GlassThickness        : float = 0.0    # 0.045; 0.1094; Half-wave plate thickness
    PCBtop_offset         : float = 0.0    # PCB top layer vertical offset from reflecting surface (after QWP fell)

    PCB_Yoffset           : float = 0
    PCB_Xoffset           : float = 0
    PCB_TrWireLengths     : float = 16     # Transport wire half length
    PCB_QuadWireLengths   : float = 62     # Quadrupole wire half length

    height_PCBLAYERTOP    : float = 0.07   # PCB Top Layer tickness (2 oz copper)
    height_PCBLAYERBOTTOM : float = 0.07   # PCB Bottom Layer tickness (2 oz copper)
    PCBCoreThickness      : float = 0.38   # 0.42; %0.38;
    # Gap between top face of Copper Z and top face of PCB (surface after QWP fell)
    CopperZ_gap           : float = 0.188 
    height_COPPERSIDEBARS : float = 1.0  # Copper Sidebars thickness
    # fmt: on

    @property
    def top_height(self):
        return self.height_PCBLAYERTOP

    @property
    def top_offset(self):
        return -(self.GlassThickness + self.PCBtop_offset + self.height_PCBLAYERTOP / 2)

    @property
    def bottom_height(self):
        return self.height_PCBLAYERBOTTOM

    @property
    def bottom_offset(self):
        # fmt: off
        return -(self.GlassThickness + self.PCBtop_offset + self.height_PCBLAYERTOP + 
                 self.PCBCoreThickness + self.height_PCBLAYERBOTTOM/2)
        # fmt: on

    @property
    def x_offset(self):
        return self.PCB_Xoffset

    @property
    def y_offset(self):
        return self.PCB_Yoffset

    @property
    def transport_wire_length(self):
        return self.PCB_TrWireLengths

    @property
    def quadrupole_wire_length(self):
        return self.PCB_QuadWireLengths

    @property
    def height(self):
        return self.height_PCBLAYERTOP + self.height_PCBLAYERBOTTOM + self.PCBCoreThickness

    @property
    def copper_z_offset(self):
        return -(self.height + self.CopperZ_gap)

    @property
    def copper_sidebars_offset(self):
        return self.copper_z_offset - self.height_COPPERSIDEBARS / 2
