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

        # Folder in which vibrations data will be saved
        self.folder = os.path.join(self.tmp_dir.name, "vibrations")

        ### Molecules

        # Slider for vibrational mode selection
        self.slider_mode_description = widgets.HTMLMath(
            r"Vibrational mode", layout=self.layout_description
        )
        # Here we don't use IntSlide to show also the total number of rotations
        self.slider_mode = widgets.SelectionSlider(
            options=["1/1"],
            value="1/1",
            # description="Vibrational mode",
            style={"description_width": "initial"},
            continuous_update=False,
            layout=self.layout,
            readout=True,
        )
        self.slider_mode.observe(self.modify_molecule, "value")

        # Molecular vibration amplitude
        self.slider_amplitude_description = widgets.HTMLMath(
            r"Temperature [K]", layout=self.layout_description
        )
        self.slider_amplitude = widgets.FloatSlider(
            value=300,
            min=10,
            max=1000,
            step=10,
            continuous_update=False,
            layout=self.layout,
        )
        self.slider_amplitude.observe(self.modify_molecule, "value")

        # Molecule selection
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
            disabled=False,
            layout=self.layout,
        )
        self.dropdown_molecule.observe(self.change_molecule, "value")

        # Toggle advanced molecule list
        self.button_advanced_molecule = widgets.Checkbox(
            value=False, indent=False, layout=Layout(width="50px")
        )
        self.button_advanced_molecule_description = widgets.HTMLMath(
            r"Extended list", layout=Layout(width="100px")
        )
        self.button_advanced_molecule.observe(self.on_molecule_list_change, "value")

        ### Appearance
        # Modify bond aspect ratio
        self.slider_aspect_ratio = widgets.FloatSlider(
            value=2, min=1, max=3, step=0.1, layout=self.layout
        )
        self.slider_aspect_ratio_description = HTMLMath(
            r"Aspect ratio", layout=self.layout_description
        )
        self.slider_aspect_ratio.observe(self.modify_representation, "value")

        # Toggle atom label
        self.tick_box_label_description=widgets.HTMLMath(value=r'Atoms label',layout=self.layout_description)
        self.tick_box_label=widgets.Checkbox(value=False,layout=self.layout)
        self.tick_box_label.observe(self.on_label_change,"value")
        # To output summary table
        self.output_summary = widgets.Output()
        self.summary_energy=HTMLMath(
            value="0",
            layout=Layout(
                width="130px", display="flex", justify_content="flex-end"
            ),
        )
        self.summary_frequency=HTMLMath(
                    value="0",
                    layout=Layout(
                        width="130px", display="flex", justify_content="flex-end"
                    ),
                )

        self.simple_molecules = [
            "O\u2082",
            "N\u2082",
            "OH",
            "H\u2082O",
            "CO\u2082"
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

        self.molecule_name = "O\u2082"
        self.representation = "ball+stick"

    def addArrows(self, *args):
        '''
        Function to add arrow in the NGLViewer showing atoms displacements
        '''
        # Need to remove arrows first to avoid visual glitch
        self.removeArrows()

        # Get the atom position to initialize arrows at correct position
        positions = list(self.traj[0].get_positions().flatten())

        n_atoms = int(len(positions) / 3)
        color = n_atoms * [0, 1, 0]
        radius = n_atoms * [0.1]

        # Initialize arrows
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

        # Get the highest amplitude to normalize all amplitudes
        scaling_factor = np.max(np.linalg.norm(self.steps[:, :, :, :], axis=2))

        def on_frame_change(change):
            '''
            Compute the new arrow position and orientations
            '''
            frame = change["new"]

            positions = self.traj[frame].get_positions()
            # Head of arrow position
            positions2 = (
                positions
                + self.steps[:, :, :, frame].reshape(-1, 3)
                / scaling_factor
                * self.slider_amp_arrow.value
            )
            # JavaScript only reads lists from Python
            positions = list(positions.flatten())
            positions2 = list(positions2.flatten())

            radius = n_atoms * [self.slider_arrow_radius.value]
            # Update the arrows position
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

        self.view.observe(on_frame_change, names=["frame"])
        # Keep in memory callable function to remove it later on
        self.handler.append(on_frame_change)

    def change_molecule(self, *args):
        """
        Compute vibrational properties of molecule \n
        Extract vibrational modes from rotations and translations
        """
        # Translate unicode characters from molecule to number
        self.molecule_name = ""
        for x in self.dropdown_molecule.value:
            if ord(x) > 128:
                # Retrieve number from unicode
                self.molecule_name += chr(ord(x) - 8320 + 48)
            else:
                self.molecule_name += x

        

        # Initialize vibration
        atoms = molecule(self.molecule_name, calculator=EMT())
        opt = BFGS(atoms, logfile=None)
        opt.run(fmax=0.001)
        vibname = os.path.join(self.folder, self.molecule_name)
        vib = Vibrations(atoms, name=vibname)
        vib.run()

        # Extract rotational motions
        ndofs = 3 * len(atoms)
        # Check if diatomic or linear molecule
        is_not_linear = int(not (len(atoms) == 2 or atoms.get_angle(0, 1, 2) == 0)) 
        nrotations = ndofs - 5 - is_not_linear

        # Get the nrotations-largest energies, to eliminate translation and rotation energies
        energies = np.absolute(vib.get_energies())
        frequencies = np.absolute(vib.get_frequencies())
        self.idx = np.argpartition(energies, -nrotations)[-nrotations:]
        self.energies = energies[self.idx]
        self.frequencies = frequencies[self.idx]
        options = []
        max_val = len(self.idx)
        for i in range(1, max_val + 1):
            options.append(str(i) + "/" + str(max_val))

        # Update available vibrational modes
        self.slider_mode.options = options

        # Write currently selected mode vibration
        mode = int(self.slider_mode.value[0])
        T = self.slider_amplitude.value
        vib.write_mode(n=self.idx[mode - 1], kT=kB * T, nimages=60)

        # Get new trajectory
        traj = Trajectory(
            os.path.join(
                self.folder,
                self.molecule_name + "." + str(self.idx[mode - 1]) + ".traj",
            )
        )
        self.replace_trajectory(traj=traj)

        # Matrix to save the atoms displacements to compute arrows
        self.steps = np.zeros((len(traj[0].positions), 1, 3, 60))
        for frame in range(len(traj)):
            step = traj[frame].positions - traj[0].positions
            self.steps[:, 0, :, frame] = step

        self.update_summary()

        # To fix advanced molecules looking weird at first
        # We update the view to reload it
        if self.button_advanced_molecule.value==True:
            self.slider_mode.value=self.slider_mode.options[-1]
            self.slider_mode.value=self.slider_mode.options[0]

    def modify_molecule(self, *args):
        """
        Rewrite vibrational mode with new temperature\n
        |!\ Here we don't have to run the vibration again
        as it has been computed before
        """
        # Initialize vibration
        atoms = molecule(self.molecule_name, calculator=EMT())
        vibname = os.path.join(self.folder, self.molecule_name)
        vib = Vibrations(atoms, name=vibname)

        # Select which mode to compute
        mode = int(self.slider_mode.value[0])
        
        T = self.slider_amplitude.value
        # Compute new vibration at different temperature
        vib.write_mode(n=self.idx[mode - 1], kT=kB * T, nimages=60)

        traj = Trajectory(
            os.path.join(
                self.folder,
                self.molecule_name + "." + str(self.idx[mode - 1]) + ".traj",
            )
        )

        # Matrix to save the atoms displacements to compute arrows
        self.steps = np.zeros((len(traj[0].positions), 1, 3, 60))
        for frame in range(len(traj)):
            step = traj[frame].positions - traj[0].positions
            self.steps[:, 0, :, frame] = step

        self.replace_trajectory(traj=traj)

        # Update frequency and energy values
        self.update_summary()

    def on_molecule_list_change(self, *args):
        '''
        Update molecule list choice
        '''
        if self.button_advanced_molecule.value == True:
            # To avoid molecule change on tick
            # We take current molecule name
            # And put it at beginning of list
            molecule = self.dropdown_molecule.value
            idx = np.argwhere(np.array(self.advanced_molecules) == molecule)
            self.advanced_molecules.pop(int(idx)) # To avoid duplication of name
            self.advanced_molecules.insert(0, molecule)

            self.dropdown_molecule.options = self.advanced_molecules

        else:
            self.dropdown_molecule.options = self.simple_molecules

    def on_label_change(self,*args):
        # https://github.com/nglviewer/nglview/issues/650
        '''
        Toggle atom label \n
        '''
        if self.tick_box_label.value ==True:
            self.view.add_label(radius=1.5,label_type='atomname',color='black',opacity=0.7,attachment="middle-center")
        else:
            self.view.remove_label()

    def print_summary(self, *args):
        """
        Print out table of energy and frequency of vibrabtion
        """
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
                    self.summary_energy,
                    self.summary_frequency
                    
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

    def update_summary(self,*args):
        """
        Update value of energy and frequency shown
        """
        mode = int(self.slider_mode.value[0])
        self.summary_energy.value=f"{self.energies[mode-1]:.3g}"
        self.summary_frequency.value=f"{self.frequencies[mode-1]:>.0f}"
