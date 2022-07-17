import numpy as np
import os

import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from ase import Atoms
from ase.io.trajectory import Trajectory
from sympy import *
from NGLUtilsClass import NGLWidgets


class NGLTrajectory(NGLWidgets):
    def __init__(self, trajectory):
        super().__init__(trajectory)

        # Atomic displacement amplitude
        self.slider_amplitude_description = widgets.HTMLMath(
            r"Oscillations amplitude", layout=self.layout_description
        )
        self.slider_amplitude = widgets.FloatSlider(
            value=0.12,
            min=0.01,
            max=0.24,
            step=0.01,
            # description="Amplitude",
            continuous_update=False,
            layout=self.layout,
        )
        self.slider_amplitude.observe(self.compute_trajectory_1D, "value")

        # Mass ratio
        self.slider_M_description = widgets.HTMLMath(
            r"$\frac{\text{M}_{\text{red}}}{\text{M}_{\text{grey}}}$",
            layout=self.layout_description,
        )
        self.slider_M = widgets.FloatSlider(
            value=2, min=1, max=5, step=0.1, continuous_update=False, layout=self.layout
        )
        self.slider_M.observe(self.compute_trajectory_1D, "value")
        self.slider_M.observe(self.update_band_M, "value")
        
        # Force constant
        self.slider_C = widgets.FloatSlider(
            value=1,
            min=0.5,
            max=5,
            step=0.1,
            description="Force constant",
            continuous_update=False,
        )

        # Chain type
        self.button_chain_description = widgets.HTMLMath(
            r"Atomic chain type", layout=self.layout_description
        )
        self.button_chain = widgets.RadioButtons(
            options=["monoatomic", "diatomic"],
            value="monoatomic",
            disabled=False,
        )
        self.button_chain.observe(self.compute_dispersion, "value")
        self.button_chain.observe(self.compute_trajectory_1D, "value")
        self.button_chain.observe(self.band_dispersion, "value")
        self.button_chain.observe(self.show_slider_M, "value")

        # Output to show slider_M
        self.output_ratio = widgets.Output()

        # Point is initialized at (0,0), and in acoustic mode
        self.x = 0
        self.y = 0
        self.ka = 0
        self.ka_array = np.linspace(-2 * np.pi, 2 * np.pi, 101)
        self.idx = 50 # idx corresponding to ka=0
        self.optic = False

        self.init_delay = 20

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
                * self.slider_amp_arrow.value
                * 10
            )
            # JavaScript only reads lists from Python
            positions = list(positions.flatten())
            positions2 = list(positions2.flatten())

            n_atoms = int(len(positions) / 3)

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

    def compute_dispersion(self, *args):
        """
        Compute the dynamical equations for the monoatomic and diatomic chains.
        For simplicity, the equation has already been simplified by factoring out the terms,
        such as \exp(-i\omega t) or \exp(k.r), with r the vector defining the atom of interest position.
        """
        if self.button_chain.value == "monoatomic":
            # Reciprocal space vector, amplitude of atom, frequency
            kx, u, w = symbols("k_x u w")
            k=Matrix([kx])
            # Mass and force constant
            M, C = symbols("M C")
            
            # Nearest neighboors assuming a=1
            atom_positions = Matrix([[-1], [1]]) #*a

            #Compute the forces acting on the atom
            # M d2u/dx2=RHS
            RHS = 0
            for i in range(atom_positions.rows):
                position = atom_positions.row(i)
                RHS += C*(exp(I * (k.T.dot(position)))*u - u)

            # Compute m*d2u/dx2
            LHS = -M * w**2 * u
            # Equation to solve : Eq=0
            Eq = RHS - LHS
            # Solve for w
            sols = solve(Eq, w)
            Sol = sols[0]

            # Create function to compute w on the fly
            self.w_ = lambdify((kx, C), Sol.subs({M: 1}))

            self.compute_dispersion_relation()

        elif self.button_chain.value == "diatomic":
            # Reciprocal space vector
            kx = symbols("k_x")
            k=Matrix([kx])
            # u is amplitude of atom 1, v is amplitude of atom 2
            u, v = symbols("u v")
            # Masses and force constant
            M1, M2, C = symbols("M_1 M_2 C")

            # Nearest neighbour distance for atom 1 and atom 2
            # The nearest neighbours are 1/2 a cell away for each atom
            atom_positions_frst_neigh_1 = Matrix([[-1/2], [1/2]])
            atom_positions_frst_neigh_2 = Matrix([[-1/2], [1/2]])

            #Compute the forces acting on the atom
            # RHS1 is for atom 1 : M1 d2u/dx2=RHS1
            # RHS2 is for atom 2 : M2 d2v/dx2=RHS2  
            RHS1 = 0 
            RHS2 = 0
            for i in range(atom_positions_frst_neigh_1.rows):
                position = atom_positions_frst_neigh_1.row(i)
                RHS1 += exp(I * (k.T.dot(position)))*v - u

            for i in range(atom_positions_frst_neigh_2.rows):
                position = atom_positions_frst_neigh_2.row(i)
                RHS2 += exp(I * (k.T.dot(position)))*u - v

            # Divide by M from left hand side
            RHS1 *= -C / M1
            RHS2 *= -C / M2
            # # Set up the dynamical matrix
            matrix = linear_eq_to_matrix([RHS1, RHS2], [u, v])[0]
            matrix.simplify()
            # Returns the eigenvalues and eigenvectors
            eig1, eig2 = matrix.eigenvects()

            # Create function to compute amplitudes and frequencies on the fly for a given k
            # /!\ We get the frequency squared as eigenvalue
            self.A_1 = lambdify((kx, M2, C), eig1[2][0][0].subs({M1: 1}))
            self.w2_1 = lambdify((kx, M2, C), eig1[0].subs({M1: 1}))

            self.A_2 = lambdify((kx, M2, C), eig2[2][0][0].subs({M1: 1}))
            self.w2_2 = lambdify((kx, M2, C), eig2[0].subs({M1: 1}))


            self.compute_dispersion_relation()

    def compute_dispersion_relation(self, *args):
        """
        Compute the correspong frequencies for a given k_array from the dispersion relation computed in "compute_dispersion" function
        For the diatomic chain, the frequency are ordered such that the optical mode has highest frequency.
        """
        if self.button_chain.value == "monoatomic":
            self.w = self.w_(self.ka_array, self.slider_C.value)

        elif self.button_chain.value == "diatomic":
            A_1 = self.A_1(self.ka_array, self.slider_M.value, self.slider_C.value)
            A_2 = self.A_2(self.ka_array, self.slider_M.value, self.slider_C.value)

            w_1 = np.sqrt(
                self.w2_1(self.ka_array, self.slider_M.value, self.slider_C.value)
            )
            w_2 = np.sqrt(
                self.w2_2(self.ka_array, self.slider_M.value, self.slider_C.value)
            )

            # Move all the highest frequencies in w_2 (optical branch)
            for i in range(len(w_1)):
                if w_1[i] > w_2[i]:
                    w_1[i], w_2[i], A_1[i], A_2[i] = w_2[i], w_1[i], A_2[i], A_1[i]

            self.w_ac = w_1
            self.w_op = w_2
            self.A_ac = A_1
            self.A_op = A_2

    def compute_trajectory_1D(self, *args):
        '''
        Compute the atom displacements for the NGLViewer for a given k.
        Compute it such that the animations loops perfectly
        '''

        # Get the k value corresponding to the one selected on the plot
        self.ka = self.ka_array[self.idx]

        # Initialize trajectory file
        traj = Trajectory(os.path.join(self.tmp_dir.name, "atoms_1d.traj"), "w")

        if self.button_chain.value == "monoatomic":
            
            # Get the frequency corresponding to the one selected on plot
            w = self.w[self.idx]
            # NGLView only reads 3D vectors
            k = np.array([self.ka, 0, 0])
            ax = np.array([1, 0, 0])

            n_frames=51
            n_atoms=20
            t_end=50
            # Matrix to save the atoms displacements to compute arrows
            self.steps = np.zeros((20, 1, 3, n_frames))
            for frame in np.linspace(0, t_end, n_frames):
                atom_positions = []
                # If frequency is non-zero, compute an effective time such that the animation loops
                if w != 0:
                    t = 2 * np.pi / t_end / np.real(w) * frame
                else:  # If frequency is zero, we fix atoms, to avoid pure translation (and division by zero)
                    t = 0

                for i in range(n_atoms):
                    # Take the real part of the displacement
                    # Displacement is given by \exp(i*(k*position-i*w*t))
                    
                    position= i * ax
                    step = (
                        np.real(
                            self.slider_amplitude.value
                            * np.exp(1j * w * t)
                            * np.exp(1j * np.dot(k, position))
                        )
                        * ax
                    )
                   
                    atom_positions_ = position + step
                    atom_positions.append(atom_positions_)
                    # Save atoms displacement
                    self.steps[i, 0, :, int(frame)] += step

                atoms = Atoms(n_atoms * "C", positions=atom_positions)
                # Save atom positions
                traj.write(atoms)

        elif self.button_chain.value == "diatomic":
            # Select the amplitude and frequency corresponding to the right branch
            if self.optic:
                self.w = self.w_op[self.idx]
                self.A = self.A_op[self.idx]
            else:
                self.w = self.w_ac[self.idx]
                self.A = self.A_ac[self.idx]

            # self.A give the ratio between atom1 and atom2 amplitude
            amp_vector = np.array([self.A, 1])
            # Normalize amplitude to avoid "infinite" displacements
            amp_vector = amp_vector / np.linalg.norm(amp_vector)
            amp1, amp2 = amp_vector[0], amp_vector[1]
            # NGLView only reads 3D vectors
            ax = np.array([1, 0, 0])
            k = np.array([self.ka, 0, 0])

            n_frames=51
            t_end=50
            # Matrix to save the atoms displacements to compute arrows
            self.steps = np.zeros((20, 1, 3, n_frames))
            for frame in np.linspace(0, t_end, n_frames):
                atom_positions = []
                if self.w != 0:
                    t = 2 * np.pi / t_end / np.real(self.w) * frame
                else: # Avoid division by zero, if w=0 atoms only translate
                    t = 0
                for i in range(10):
                    position= i * ax 
                    # Compute and take real part of displacement for atom 1
                    # Displacement is given by \exp(i*(k*position-i*w*t))
                    step = (
                        np.real(
                            self.slider_amplitude.value
                            * amp1
                            * np.exp(-1j * self.w * t)
                            * np.exp(1j * np.dot(k, position))
                        )
                        * ax
                    )
                    # Compute new position at time t
                    # We multiplit position by 2 for visualization purposes
                    atom_positions_1 = position*2 + step 
                    # Save atom 1 step in even number index
                    self.steps[2 * i, 0, :, int(frame)] += step

                    # Compute and take real part of displacement for atom 1
                    # Displacement is given by \exp(i*(k*position-i*w*t))
                    # Atom 2 is positioned half a cell further away from atom 1
                    # which explains (i+1/2)a
                    position= (i * ax + 1/2 * ax )
                    step = (
                        np.real(
                            self.slider_amplitude.value
                            * amp2
                            * np.exp(-1j * self.w * t)
                            * np.exp(1j * np.dot(k, position))
                        )
                        * ax
                    )
                    # Compute new position at time t
                    # We multiplit position by 2 for visualization purposes
                    atom_positions_2 = position*2 + step
                    # # Save atom 2 step in odd number index
                    self.steps[2 * i + 1, 0, :, int(frame)] += step

                    atom_positions.append(atom_positions_1)
                    atom_positions.append(atom_positions_2)

                atoms = Atoms(
                    int(len(atom_positions) / 2) * "CO", positions=atom_positions
                )
                # Save new atoms position
                traj.write(atoms)
        # For the Trajectory to be read by NGLView, we need to open it in read mode
        new_traj = Trajectory(os.path.join(self.tmp_dir.name, "atoms_1d.traj"))
        # Update the trajectory in NGLView
        self.replace_trajectory(traj=new_traj, representation="spacefill")
        self.view.center()
        self.view.control.zoom(0.25)

    def initialize_dispersion_plot(self, *args):
        '''
        Initialize the dispersion plot axes and curves color/style
        
        '''
        plt.ioff()
        px = 1 / plt.rcParams["figure.dpi"]
        self.fig, self.ax = plt.subplots(figsize=(500 * px, 300 * px))

        # Remove toolbar, header and footer of the plot to make it clean
        self.fig.canvas.toolbar_visible = False
        self.fig.canvas.header_visible = False
        self.fig.canvas.footer_visible = False

        # Dispersion lines inside first BZ
        (self.lines_ac,) = self.ax.plot([], [], c="blue")
        (self.lines_op,) = self.ax.plot([], [], c="orange")
        (self.lines,) = self.ax.plot([], [], c="blue")

        # Dispersion lines outside first BZ
        (self.lines_ac_out,) = self.ax.plot([], [], "--", c="blue", alpha=0.5)
        (self.lines_op_out,) = self.ax.plot([], [], "--", c="orange", alpha=0.5)
        (self.lines_out,) = self.ax.plot([], [], "--", c="blue", alpha=0.5)

        # Selected frequency point
        (self.point,) = self.ax.plot([], [], ".", c="crimson", markersize=15)

        self.ax.set_xlabel("k")
        self.ax.set_ylabel("$\omega$")
        self.fig.set_tight_layout(tight=True)
        plt.ion()

    def band_dispersion(self, *args):
        """
        Update the band dispersion plot upon change of chain type (monoatomic/diatomic)
        """
        # Get size of a pixel
        px = 1 / plt.rcParams["figure.dpi"]
        plt.ioff()

        if self.button_chain.value == "monoatomic":
            # Reset figure width to initial width
            self.fig.set_figwidth(500 * px)
            self.ax.set_xticks(np.linspace(-np.pi, np.pi, 5))
            self.ax.set_xticklabels(
                [r"$-\frac{\pi}{a}$", "", "0", "", r"$\frac{\pi}{a}$"]
            )
            self.lines_out.set_data((self.ka_array, self.w))
            # Here we want to only have the line inside the BZ to be full
            self.lines.set_data((self.ka_array[25:75], self.w[25:75]))

            # Remove diatomic chain lines
            self.lines_ac.set_data(([], []))
            self.lines_op.set_data(([], []))
            self.lines_ac_out.set_data(([], []))
            self.lines_op_out.set_data(([], []))

        elif self.button_chain.value == "diatomic":
            # Reduce figure width to "half" the size, since x axis is half as big now.
            self.fig.set_figwidth(300 * px)
            # To make life easier, we consider the diatomic lattice parameter
            # to be the same as the monoatomic lattice parameter
            # But for better comprehension, we label that the new lattice parameter
            # is twice the monoatomic one
            self.ax.set_xticks(np.linspace(-np.pi, np.pi, 5))
            self.ax.set_xticklabels(
                [r"$-\frac{\pi}{2a}$", "", "0", "", r"$\frac{\pi}{2a}$"]
            )
            self.ax.set_ylabel("$\omega$")
            self.lines_ac_out.set_data((self.ka_array, self.w_ac))
            self.lines_op_out.set_data((self.ka_array, self.w_op))
            # Here we want to only have the line inside the BZ to be full
            self.lines_ac.set_data((self.ka_array[25:75], self.w_ac[25:75]))
            self.lines_op.set_data((self.ka_array[25:75], self.w_op[25:75]))

            # Remove monoatomic chain lines
            self.lines.set_data(([], []))
            self.lines_out.set_data(([], []))

        # First BZ limit
        self.ax.plot([-np.pi, -np.pi], [0, 2.2], "k--", linewidth=1)
        self.ax.plot([np.pi, np.pi], [0, 2.2], "k--", linewidth=1)

        # Set the plot bounds to array bounds
        self.ax.set_xbound((-2 * np.pi), (2 * np.pi))
        # 2.2 works well for initial height
        self.ax.set_ybound(0, 2.2)

        self.point.set_data((0, 0))
        self.fig.canvas.mpl_connect("button_press_event", self.onclick)
        plt.ion()

    def onclick(self, event):
        """
        Determine frequency and k point upon click on band dispersion figure
        """
        self.x = event.xdata
        self.y = event.ydata

        # Get the idx of the closest k in the array
        self.idx = (np.abs(self.ka_array - self.x)).argmin()
        self.ka = self.ka_array[self.idx]

        if self.button_chain.value == "monoatomic":
            w = self.w[self.idx]
        elif self.button_chain.value == "diatomic":
            # Compute if acoustic or optical
            w = self.compute_distance(y=self.y)

        # Update point position
        self.point.set_data(self.ka, w)
        self.compute_trajectory_1D()

    def compute_distance(self, y):
        """
        Compute vertical distance between point and acoustic/optical branches
        And return corresponding frequency
        """
        if np.abs((y - self.w_op[self.idx])) < np.abs((y - self.w_ac[self.idx])):
            self.optic = True
            return self.w_op[self.idx]
        else:
            self.optic = False
            return self.w_ac[self.idx]

    def update_band_M(self, *args):
        """
        Recompute band dispersion upon mass ratio update
        """
        self.compute_dispersion_relation()

        # Re-adjust frequency to closest one
        self.idx = (np.abs(self.ka_array - self.x)).argmin()
        self.ka = self.ka_array[self.idx]

        # Get the frequency according to optical or acoustic mode
        w = self.compute_distance(y=self.y)

        self.point.set_data(self.ka, w)
        # Update the trajectory
        self.compute_trajectory_1D()
        # Update band dispersion lines
        self.lines_ac.set_data((self.ka_array[25:75], self.w_ac[25:75]))
        self.lines_op.set_data((self.ka_array[25:75], self.w_op[25:75]))
        self.lines_ac_out.set_data((self.ka_array, self.w_ac))
        self.lines_op_out.set_data((self.ka_array, self.w_op))

    def show_slider_M(self, *args):
        """
        Show mass slider if diatomic chain selected
        """
        if self.button_chain.value == "monoatomic":
            self.output_ratio.clear_output()
        elif self.button_chain.value == "diatomic":
            with self.output_ratio:
                display(widgets.HBox([self.slider_M_description, self.slider_M]))
