import time
import numpy as np

import os

import ipywidgets as widgets
from ipywidgets import HTML, Label, HBox, VBox, IntSlider, HTMLMath, Output, Layout
from IPython.display import display
from ase.build import molecule
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from ase.vibrations import Vibrations
from ase.io.trajectory import Trajectory
from ase.units import kB

from NGLUtilsClass import NGLWidgets


class NGLMolecule(NGLWidgets):
    def __init__(self, trajectory) -> None:
        super().__init__(trajectory)

        self.output_summary = widgets.Output()
        self.folder = os.path.join(self.tmp_dir.name, "vibrations")
        # Molecules
        self.slider_mode_description = widgets.HTMLMath(
            r"Vibrational mode", layout=self.layout_description
        )
        self.slider_mode = widgets.SelectionSlider(
            options=["1/1"],
            value="1/1",
            # description="Vibrational mode",
            style={"description_width": "initial"},
            continuous_update=False,
            layout=self.layout,
            readout=True,
        )

        self.slider_amplitude_description = widgets.HTMLMath(
            r"Temperature [K]", layout=self.layout_description
        )
        self.slider_amplitude = widgets.FloatSlider(
            value=300,
            min=10,
            max=1000,
            step=10,
            # description="Temperature [K]",
            continuous_update=False,
            layout=self.layout,
        )

        self.dropdown_molecule_description = widgets.HTMLMath(
            r"Molecule", layout=self.layout_description
        )
        self.dropdown_molecule = widgets.Dropdown(
            options=[
                "O\u2082",
                "N\u2082",
                "OH",
                "H\u2082O",
                "CO\u2082",
                "CH\u2084",
                "NH\u2083",
                "C\u2086H\u2086",
            ],
            value="O\u2082",
            # description="Molecule:",
            disabled=False,
            # style={"font_weight": "bold"},
            layout=self.layout,
        )

        self.dropdown_molecule.observe(self.change_molecule, "value")
        self.slider_mode.observe(self.modify_molecule, "value")
        self.slider_amplitude.observe(self.modify_molecule, "value")

        self.representation = "ball+stick"

        self.slider_aspect_ratio = widgets.FloatSlider(
            value=2, min=1, max=3, step=0.1, layout=Layout(width="200px")
        )
        self.slider_aspect_ratio_description = HTMLMath(
            r"Aspect ratio", layout=self.layout_description
        )
        self.slider_aspect_ratio.observe(self.modify_representation, "value")

        self.simple_molecules = [
            "O\u2082",
            "N\u2082",
            "OH",
            "H\u2082O",
            "CO\u2082",
            "CH\u2084",
            "NH\u2083",
            "C\u2086H\u2086",
        ]
        self.advanced_molecules = [
            "CH\u2083CHO",
            "H\u2082COH",
            "OCHCHO",
            "C\u2083H\u2089C",
            "CH\u2083CH\u2082OCH\u2083",
            "HCOOH",
            "H\u2082",
            "C\u2082H\u2082",
            "C\u2084H\u2084NH",
            "CH\u2083CO",
            "CO",
            "C\u2082H\u2086CHOH",
            "CH\u2082NHCH\u2082",
            "HCO",
            "C\u2082H\u2086",
            "CN",
            "H\u2083CNH\u2082",
            "CH\u2083CH\u2082OH",
            "C\u2082H\u2083",
            "CH\u2083CN",
            "CH\u2083ONO",
            "CO\u2082",
            "NO",
            "NH\u2082",
            "CH",
            "CH\u2082OCH\u2082",
            "C\u2086H\u2086",
            "CH\u2083CONH\u2082",
            "H\u2082CCHCN",
            "H\u2082CO",
            "CH\u2083COOH",
            "N\u2082H\u2084",
            "OH",
            "CH\u2083OCH\u2083",
            "C\u2085H\u2085N",
            "H\u2082O",
            "CH\u2083NO\u2082",
            "C\u2084H\u2084O",
            "CH\u2083O",
            "CH\u2083OH",
            "CH\u2083CH\u2082O",
            "C\u2083H\u2087",
            "CH\u2083",
            "O\u2083",
            "C\u2082H\u2084",
            "NCCN",
            "C\u2082H\u2085",
            "N\u2082O",
            "HCN",
            "C\u2082H\u2086NH",
            "C\u2083H\u2088",
            "O\u2082",
            "NH",
            "NH\u2083",
            "C\u2083H\u2089N",
            "HCOOCH\u2083",
            "CCH",
            "C\u2085H\u2088",
            "CH\u2083COCH\u2083",
            "CH\u2084",
            "H\u2082CCO",
            "CH\u2083CH\u2082NH\u2082",
            "N\u2082",
            "H\u2082O\u2082",
            "NO\u2082",
            "C\u2087NH\u2085",
            "bicyclobutane",
            "methylenecyclopropane",
            "cyclobutene",
            "trans-butane",
            "cyclobutane",
            "butadiene",
            "isobutane",
            "BDA",
            "biphenyl",
        ]

        self.button_advanced_molecule = widgets.Checkbox(
            value=False, indent=False, layout=Layout(width="50px")
        )
        self.button_advanced_molecule_description = widgets.HTMLMath(
            r"Extended list", layout=Layout(width="100px")
        )
        self.button_advanced_molecule.observe(self.on_molecule_list_change, "value")
        self.dropdown_molecule_advanced_description = widgets.HTMLMath(
            r"Molecule", layout=self.layout_description
        )
        self.dropdown_molecule_advanced = widgets.Dropdown(
            options=self.advanced_molecules,
            value="O\u2082",
            # description="Molecule:",
            disabled=False,
            # style={"font_weight": "bold"},
            layout=self.layout,
        )
        self.dropdown_molecule_advanced.observe(self.change_molecule_advanced, "value")

        self.molecule_name = "O\u2082"

    def on_molecule_list_change(self, *args):
        if self.button_advanced_molecule.value == True:
            # To avoid molecule change on tick
            molecule = self.dropdown_molecule.value
            idx = np.argwhere(np.array(self.advanced_molecules) == molecule)
            self.advanced_molecules.pop(59)
            self.advanced_molecules.insert(0, molecule)

            self.dropdown_molecule.options = self.advanced_molecules

        else:
            self.dropdown_molecule.options = self.simple_molecules

    def addArrows(self, *args):
        self.removeArrows()

        positions = list(self.traj[0].get_positions().flatten())

        n_atoms = int(len(positions) / 3)
        color = n_atoms * [0, 1, 0]
        radius = n_atoms * [0.1]
        self.view._js(
            f"""
        var shape = new NGL.Shape("my_shape")

        var arrowBuffer = new NGL.ArrowBuffer({{position1: new Float32Array({positions}),
        position2: new Float32Array({positions}),
        color: new Float32Array({color}),
        radius: new Float32Array({radius})
        }})

        shape.addBuffer(arrowBuffer)
        globalThis.arrowBuffer = arrowBuffer;
        var shapeComp = this.stage.addComponentFromObject(shape)
        shapeComp.addRepresentation("buffer")
        shapeComp.autoView()
        """
        )
        # Remove observe callable to avoid visual glitch
        if self.handler:
            self.view.unobserve(self.handler.pop(), names=["frame"])

        scaling_factor = np.max(np.linalg.norm(self.steps[:, :, :, :], axis=2))

        def on_frame_change(change):
            frame = change["new"]

            positions = self.traj[frame].get_positions()
            positions2 = (
                positions
                + self.steps[:, :, :, frame].reshape(-1, 3)
                / scaling_factor
                * self.slider_amp_arrow.value
            )
            positions = list(positions.flatten())
            positions2 = list(positions2.flatten())

            if self.tick_box_arrows.value:
                radius = n_atoms * [self.slider_arrow_radius.value]
                self.view._js(
                    f"""
                globalThis.arrowBuffer.setAttributes({{
                position1: new Float32Array({positions}),
                position2: new Float32Array({positions2}),
                radius: new Float32Array({radius})
                }})
                
                this.stage.viewer.requestRender()
                """
                )
            else:
                radius = n_atoms * [0.0]
                self.view._js(
                    f"""
                globalThis.arrowBuffer.setAttributes({{
                position1: new Float32Array({positions}),
                position2: new Float32Array({positions}),
                radius: new Float32Array({radius})
                }})

                this.stage.viewer.requestRender()
                """
                )

        self.view.observe(on_frame_change, names=["frame"])
        self.handler.append(on_frame_change)

    def modify_molecule(self, *args):
        """
        Example slider:
            slider_amplitude=FloatSlider(value=300,min=10,max=1000,step=10,description='Temperature [K]',continuous_update=False)
            slider_amplitude.observe(functools.partial(set_amplitude,view,slider))
        """

        time.sleep(0.2)

        # self.molecule_name = ""
        # for x in self.dropdown_molecule.value:
        #     if ord(x) > 128:
        #         # Retrieve number from unicode
        #         self.molecule_name += chr(ord(x) - 8320 + 48)
        #     else:
        #         self.molecule_name += x

        atoms = molecule(self.molecule_name, calculator=EMT())
        vibname = os.path.join(self.folder, self.molecule_name)
        vib = Vibrations(atoms, name=vibname)
        # TODO : write in temporary file

        mode = int(self.slider_mode.value[0])
        T = self.slider_amplitude.value
        vib.write_mode(n=self.idx[mode - 1], kT=kB * T, nimages=60)

        traj = Trajectory(
            os.path.join(
                self.folder,
                self.molecule_name + "." + str(self.idx[mode - 1]) + ".traj",
            )
        )

        self.steps = np.zeros((len(traj[0].positions), 1, 3, 60))

        for frame in range(len(traj)):
            step = traj[frame].positions - traj[0].positions
            self.steps[:, 0, :, frame] = step

        self.replace_trajectory(traj=traj)

        self.print_summary()

    def change_molecule(self, *args):
        time.sleep(0.2)  # Time to get the dropdown value to update
        self.molecule_name = ""
        for x in self.dropdown_molecule.value:
            if ord(x) > 128:
                # Retrieve number from unicode
                self.molecule_name += chr(ord(x) - 8320 + 48)
            else:
                self.molecule_name += x

        atoms = molecule(self.molecule_name, calculator=EMT())

        # Relax and get vibrational properties
        opt = BFGS(atoms, logfile=None)
        opt.run(fmax=0.001)

        vibname = os.path.join(self.folder, self.molecule_name)
        vib = Vibrations(atoms, name=vibname)
        vib.run()

        # Extract rotational motions
        ndofs = 3 * len(atoms)
        is_not_linear = int(not (len(atoms) == 2 or atoms.get_angle(0, 1, 2) == 0))
        nrotations = ndofs - 5 - is_not_linear
        energies = np.absolute(vib.get_energies())
        frequencies = np.absolute(vib.get_frequencies())

        # Get the nrotations-largest energies, to eliminate translation and rotation energies
        self.idx = np.argpartition(energies, -nrotations)[-nrotations:]
        self.energies = energies[self.idx]
        self.frequencies = frequencies[self.idx]
        options = []
        max_val = len(self.idx)
        for i in range(1, max_val + 1):
            options.append(str(i) + "/" + str(max_val))
        # self.slider_mode.max = len(self.idx) - 1
        self.slider_mode.options = options

        mode = int(self.slider_mode.value[0])

        # TODO : write in temporary file
        T = self.slider_amplitude.value
        vib.write_mode(n=self.idx[mode - 1], kT=kB * T, nimages=60)

        traj = Trajectory(
            os.path.join(
                self.folder,
                self.molecule_name + "." + str(self.idx[mode - 1]) + ".traj",
            )
        )
        self.steps = np.zeros((len(traj[0].positions), 1, 3, 60))

        for frame in range(len(traj)):
            step = traj[frame].positions - traj[0].positions
            self.steps[:, 0, :, frame] = step

        self.replace_trajectory(traj=traj)

        self.print_summary()

    def change_molecule_advanced(self, *args):
        time.sleep(0.2)  # Time to get the dropdown value to update
        self.molecule_name = ""
        for x in self.dropdown_molecule_advanced.value:
            if ord(x) > 128:
                # Retrieve number from unicode
                self.molecule_name += chr(ord(x) - 8320 + 48)
            else:
                self.molecule_name += x

        atoms = molecule(self.molecule_name, calculator=EMT())

        # Relax and get vibrational properties
        opt = BFGS(atoms, logfile=None)
        opt.run(fmax=0.001)

        vibname = os.path.join(self.folder, self.molecule_name)
        vib = Vibrations(atoms, name=vibname)
        vib.run()

        # Extract rotational motions
        ndofs = 3 * len(atoms)
        is_not_linear = int(not (len(atoms) == 2 or atoms.get_angle(0, 1, 2) == 0))
        nrotations = ndofs - 5 - is_not_linear
        energies = np.absolute(vib.get_energies())
        frequencies = np.absolute(vib.get_frequencies())

        # Get the nrotations-largest energies, to eliminate translation and rotation energies
        self.idx = np.argpartition(energies, -nrotations)[-nrotations:]
        self.energies = energies[self.idx]
        self.frequencies = frequencies[self.idx]
        options = []
        max_val = len(self.idx)
        for i in range(1, max_val + 1):
            options.append(str(i) + "/" + str(max_val))
        # self.slider_mode.max = len(self.idx) - 1
        self.slider_mode.options = options

        mode = int(self.slider_mode.value[0])

        # TODO : write in temporary file
        T = self.slider_amplitude.value
        vib.write_mode(n=self.idx[mode - 1], kT=kB * T, nimages=60)

        traj = Trajectory(
            os.path.join(
                self.folder,
                self.molecule_name + "." + str(self.idx[mode - 1]) + ".traj",
            )
        )
        self.steps = np.zeros((len(traj[0].positions), 1, 3, 60))

        for frame in range(len(traj)):
            step = traj[frame].positions - traj[0].positions
            self.steps[:, 0, :, frame] = step

        self.replace_trajectory(traj=traj)

        self.print_summary()

    def print_summary(self, *args):
        mode = int(self.slider_mode.value[0])
        with self.output_summary:
            self.output_summary.clear_output()
            titles = HBox(
                [
                    HTMLMath(value=1 * " ", layout=Layout(width="100px")),  # Spacer
                    HTMLMath(
                        value="Energy [meV]",
                        layout=Layout(
                            width="130px", display="flex", justify_content="flex-end"
                        ),
                    ),
                    HTMLMath(
                        value=r"Frequency [cm$^{-1}$]",
                        layout=Layout(
                            width="130px", display="flex", justify_content="flex-end"
                        ),
                    ),
                ]
            )
            values = HBox(
                [
                    HTMLMath(value=1 * " ", layout=Layout(width="100px")),  # Spacer
                    HTMLMath(
                        value=f"{self.energies[mode-1]:>12.3f}",
                        layout=Layout(
                            width="130px", display="flex", justify_content="flex-end"
                        ),
                    ),
                    HTMLMath(
                        value=f"{self.frequencies[mode-1]:>12.0f}",
                        layout=Layout(
                            width="130px", display="flex", justify_content="flex-end"
                        ),
                    ),
                ]
            )
            nota_bene = HBox(
                [
                    HTMLMath(value=1 * " ", layout=Layout(width="100px")),  # Spacer
                    HTMLMath(
                        value=f"N.B.: Energies and frequencies are computed using the EMT potential and do not provide accurate values.",
                        layout=Layout(width="270px"),
                    ),
                ]
            )
            display(titles, values, nota_bene)
            # print(f"{'Energy [meV]':>15}{'Frequency [cm$^-{1}$]':>25}")
            # print(
            #     f"{self.energies[mode-1]:>15.3f}{self.frequencies[mode-1]:>25.0f}"
            # )

    def change_mode(self, *args):
        mode = int(self.slider_mode.value[0])
        # molecule_name = self.molecule_name
        traj = Trajectory(
            os.path.join(
                self.folder,
                self.molecule_name + "." + str(self.idx[mode - 1]) + ".traj",
            )
        )
        self.replace_trajectory(traj=traj)
