import nglview as nv
import threading
import moviepy.editor as mpy
import time
import tempfile
import numpy as np
import functools
import os
import json

import ipywidgets as widgets
from ipywidgets import HBox, VBox, HTMLMath, Output, Layout


from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
from IPython.display import Image


class NGLWidgets:
    def __init__(self, trajectory) -> None:

        # Simulation folder
        self.tmp_dir = tempfile.TemporaryDirectory(prefix="user_", dir=".")

        # Initialize view with trajectory
        self.traj = trajectory
        self.view = nv.show_asetraj(self.traj)

        self.zoom = 15

        # Layouts for all widgets
        self.layout_description = widgets.Layout(width="130px")
        self.layout = Layout(width="200px")

        # Camera
        layout_camera = widgets.Layout(width="50px")
        style = {"description_width": "initial"}
        self.button_x = widgets.Button(description="x", layout=layout_camera)
        self.button_x.on_click(functools.partial(self.set_camera, direction="x"))
        self.button_y = widgets.Button(description="y", layout=layout_camera)
        self.button_y.on_click(functools.partial(self.set_camera, direction="y"))
        self.button_z = widgets.Button(description="z", layout=layout_camera)
        self.button_z.on_click(functools.partial(self.set_camera, direction="z"))

        # Outputs
        self.output_text = widgets.Output()
        self.output_movie = widgets.Output()
        self.output_camera = widgets.Output()
        self.output_gif = widgets.Output()
        self.output_camera_position = Output()
        self.output_camera_position_error = Output()


        # Custom camera orientation
        self.view.observe(self.on_orientation_change, names=["_camera_orientation"])
        self.camera_orientation_description = HTMLMath(
            r"Camera orientation :", layout=self.layout_description
        )
        self.text_orientation = widgets.Textarea(value="Paste camera orientation here")
        self.text_orientation.observe(self.change_camera_position, names=["value"])

        # Movie
        self.dropdown_resolution_description = HTMLMath(
            r"Resolution", layout=self.layout_description
        )
        self.dropdown_resolution = widgets.Dropdown(
            options=["480p", "720p", "1080p", "1440p", "2K", "500x500p", "1000x1000p"],
            value="1080p",
            disabled=False,
            layout=self.layout,
        )

        self.slider_speed_description = HTMLMath(
            r"Animation speed", layout=self.layout_description
        )
        self.slider_speed = widgets.FloatSlider(
            value=1,
            min=0.1,
            max=2,
            step=0.1,
            continuous_update=False,
            layout=self.layout,
        )
        self.slider_speed.observe(self.set_speed, "value")

        self.button_movie = widgets.Button(
            description="Render GIF", style=style, layout=self.layout_description
        )
        self.button_movie.on_click(self.make_movie)

        # Appearance
        self.slider_amp_arrow_description = HTMLMath(
            r"Arrow amplitude", layout=self.layout_description
        )
        self.slider_amp_arrow = widgets.FloatSlider(
            value=2.0,
            min=0.1,
            max=5.01,
            step=0.1,
            continuous_update=False,
            layout=self.layout,
        )

        self.tick_box_arrows_description = HTMLMath(
            r"Show arrows", layout=self.layout_description
        )
        self.tick_box_arrows = widgets.Checkbox(
            value=False,
        )
        self.tick_box_arrows.observe(self.on_arrows_change,"value")

        self.slider_arrow_radius_description = HTMLMath(
            r"Arrow radius", layout=self.layout_description
        )
        self.slider_arrow_radius = widgets.FloatSlider(
            value=0.2,
            min=0.01,
            max=0.3,
            step=0.01,
            continuous_update=False,
            layout=self.layout,
        )

        self.slider_atom_radius_description = HTMLMath(
            r"Atom radius", layout=self.layout_description
        )
        self.slider_atom_radius = widgets.FloatSlider(
            value=0.2,
            min=0.01,
            max=1,
            step=0.01,
            continuous_update=False,
            layout=self.layout,
        )
        self.slider_atom_radius.observe(self.modify_representation, "value")

        # Premade layouts
        self.camera = widgets.VBox(
            [
                widgets.Label(value="$\Large{\\text{Camera view}}$"),
                widgets.HBox([self.button_x, self.button_y, self.button_z]),
            ]
        )

        self.arrow = widgets.VBox(
            [
                HBox([self.slider_amp_arrow_description, self.slider_amp_arrow]),
                HBox([self.slider_arrow_radius_description, self.slider_arrow_radius]),
                HBox([self.tick_box_arrows_description, self.tick_box_arrows]),
            ]
        )

        self.movie = VBox(
            [
                HBox([self.button_movie]),
                HBox([self.dropdown_resolution_description, self.dropdown_resolution]),
                HBox([self.slider_speed_description, self.slider_speed]),
                self.output_movie,
            ]
        )

        # Widget list to disable when rendering GIF
        self.widgetList = [
            self.button_x,
            self.button_y,
            self.button_z,
            self.button_movie,
            self.slider_speed,
            self.slider_amp_arrow,
            self.slider_arrow_radius,
            self.slider_atom_radius,
            self.tick_box_arrows,
            self.dropdown_resolution,
            self.text_orientation,
        ]

        # Set delay for animation
        self.init_delay = 25
        self.tmpFileName_movie = None
        # Keep in memory view handlers
        self.handler = []
        # Set base representation
        self.representation = "spacefill"

    def replace_trajectory(self, *args, traj, representation="ball+stick"):
        """
        Update view with new trajectory
        """
        self.traj = traj
        comp_ids = []
        orientation_ = self.view._camera_orientation
        orientation = [x for x in orientation_]

        # # Camera view is empty before viewing
        # if orientation:
        #     # Keep last number to 1, as it is last number in rotation matrix
        #     orientation.pop()
        #     orientation.append(1)

        # Need to remove arrows or new trajectory will not move
        self.removeArrows()

        # Remove all components except the newly added one
        for comp_id in self.view._ngl_component_ids:
            comp_ids.append(comp_id)

        self.view.add_trajectory(self.traj)

        # Once new view is added, we can remove old components
        for comp_id in comp_ids:
            self.view.remove_component(comp_id)
        
        # self.view._set_camera_orientation(orientation)
        # Modify atoms representation - spacefill is default one
        self.modify_representation()

        # Add arrows if tick box checked
        if self.tick_box_arrows.value == True:
            self.addArrows()

    def modify_representation(self, *args):
        """
        Change the view representation to either spacefill or ball+stick.\n
        Spacefill parameters : atoms radius\n
        ball+stick parameters : atoms radius, aspect ratio\n
        """
        self.view.clear_representations()
        if self.representation == "spacefill":
            self.view.add_representation(
                self.representation,
                selection="all",
                radius=self.slider_atom_radius.value,
            )
        elif self.representation == "ball+stick":
            self.view.add_representation(
                self.representation,
                selection="all",
                radius=self.slider_atom_radius.value,
                aspectRatio=self.slider_aspect_ratio.value,
            )

    def addArrows(self, *args):
        # https://github.com/nglviewer/nglview/discussions/1002
        """
        Abstract class to add arrows in animations \n
        addArrows defined in NGLTrajectory,NGLTrajectory2D and NGLMolecule \n
        """
        pass

    def removeArrows(self):
        # Adapted from https://projects.volkamerlab.org/teachopencadd/talktorials/T017_advanced_nglview_usage.html?highlight=nglview#Access-the-JavaScript-layer
        """
        Remove arrows from view
        """
        # Remove observed function from view to avoid calling multiple functions
        if self.handler:
            self.view.unobserve(self.handler.pop(), names=["frame"])

        self.view._execute_js_code(
            """
        this.stage.removeComponent(this.stage.getComponentsByName("my_shape").first)
        """
        )
    def on_arrows_change(self, *args):
        """
        Toggle arrows view
        """
        if self.tick_box_arrows.value:
            self.addArrows()
        else:
            self.removeArrows()

    def set_view_dimensions(self, width=640, height=480):
        """
        Set view width and height in px
        """
        self.view._remote_call(
            "setSize", target="Widget", args=[str(width) + "px", str(height) + "px"]
        )

    def set_player_parameters(self, **kargs):
        """
        Available parameters :
            step : Initial value = 1
                Available values [-100, 100]
            delay : Initial value = 100
                Available values : [10, 1000]
        """
        self.view.player.parameters = kargs

    def set_view_parameters(self, **kwargs):
        """
        Available parameters:
            panSpeed : Initial value = 1
                Available values : [0, 10]
            rotateSpeed : Initial value = 2
                Available values : [0, 10]
            zoomSpeed : Initial value = 1
                Available values : [0, 10]
            clipDist : Initial value = 10
                Available values : [0, 200]
            cameraFov : Initial value = 40
                Available values : [15, 120]
            clipFar : Initial value = 100
                Available values : [0, 100]
            clipNear : Initial value = 0
                Available values : [0, 100]
            fogFar : Initial value = 100
                Available values : [Available values]
            fogNear : Initial value = 50
                Available values : [0, 100]
            lightIntensity : Initial value = 1
                Available values : [0, 10]
            quality : Initial value = 'medium'
                Available values : 'low', 'medium', 'high'
            backgroundColor : Initial value = 'white'
                Available values : color name or HEX

        """
        self.view.parameters = kwargs

    def set_speed(self, *args):
        """
        Set animation speed from slider value
        """
        self.view.player.update_parameters(
            change={"new": {"delay": self.init_delay / self.slider_speed.value}}
        )

    def change_resolution(self, *args):
        """
        Change the view dimension with respect to dropdown value
        """
        # Resolution value is set to half the wanted resolution, as the downloaded GIF has twice the resolution
        if self.dropdown_resolution.value == "480p":
            self.set_view_dimensions(320, 240)
        elif self.dropdown_resolution.value == "720p":
            self.set_view_dimensions(640, 360)
        elif self.dropdown_resolution.value == "1080p":
            self.set_view_dimensions(960, 540)
        elif self.dropdown_resolution.value == "1440p":
            self.set_view_dimensions(1280, 720)
        elif self.dropdown_resolution.value == "2K":
            self.set_view_dimensions(1024, 540)
        elif self.dropdown_resolution.value == "500x500p":
            self.set_view_dimensions(250, 250)
        elif self.dropdown_resolution.value == "1000x1000p":
            self.set_view_dimensions(500, 500)

    def make_movie(self, *args):
        # Adapted from https://github.com/nglviewer/nglview/issues/928
        """
        Create a GIF of the current view animation
        """
        # Remove gif preview
        self.output_gif.outputs = ()
        # Stop animation
        self.view._iplayer.children[0]._playing = False
        # Set resolution
        self.change_resolution()
        # Start thread to compile GIF
        thread = threading.Thread(
            target=self.process_images,
        )
        thread.daemon = True
        thread.start()

    def process_images(self, *args):
        """
        Generate pictures which are later compiled into a GIF
        """
        # Disable widgets to avoid glitch in video
        for widget in self.widgetList:
            widget.disabled = True

        n_frames = self.view.max_frame + 1

        # Remove output text
        self.output_movie.outputs = ()
        self.output_movie.append_stdout("Generating GIF, please wait...")

        # Create temporary folder for pictures so that it can be deleted after GIF is compiled
        tmp_dir_frames = tempfile.TemporaryDirectory(
            prefix="frames_", dir=self.tmp_dir.name
        )
        try:
            for frame in range(n_frames):
                counter = 0
                im = self.view.render_image(frame=frame, factor=2)

                # Sleep time to give time for picture to be generated
                while not im.value:
                    time.sleep(0.1)
                    counter += 1
                    # If it takes too long, then pictures cannot be generated
                    if counter > 50:
                        self.output_movie.outputs = ()
                        self.output_movie.append_stdout("Could not generate pictures")
                        raise Exception("Could not generate pictures")
                path = os.path.join(tmp_dir_frames.name, f"frame{frame}.png")
                with open(path, "wb") as f:
                    f.write(im.value)

            # Resets view dimensions to original
            self.set_view_dimensions()

            # Compile pictures into gif
            self.compile_movie(directory=tmp_dir_frames.name, n_frames=n_frames)
            self.show_gif_preview()

        except Exception as error:
            self.output_movie.clear_output()
            self.output_movie.append_stdout(error)
        finally:
            # Reactivate widgets
            for widget in self.widgetList:
                widget.disabled = False
            # Resets view dimension to original
            self.set_view_dimensions()

        # Delete frame folder
        tmp_dir_frames.cleanup()

        self.output_movie.outputs = ()
        self.output_movie.append_stdout("Right click on GIF to download it")

    def compile_movie(self, *args, directory, n_frames):
        """
        Compile set of PNG pictures into GIF \n
        Args:\n
            directory : directory in which the pictures are located\n
            n_frames : total number of pictures\n
        """
        imagefiles = [
            os.path.join(directory, f"frame{i}.png") for i in range(0, n_frames)
        ]
        # Compute the FPS to get the same speed as animation on screen
        frame_per_second = round(1000 / (25 / self.slider_speed.value))

        im = mpy.ImageSequenceClip(imagefiles, fps=frame_per_second)
        # Create temporary file name to avoid erasing previously created movies
        with tempfile.NamedTemporaryFile(
            dir=".", prefix="movie_", suffix=".gif", delete=False
        ) as tmpFile:
            tmpFileName_movie = os.path.basename(tmpFile.name)
            self.tmpFileName_movie = os.path.join(self.tmp_dir.name, tmpFileName_movie)
            im.write_gif(
                self.tmpFileName_movie, fps=frame_per_second, verbose=False, logger=None
            )

    def show_gif_preview(self, *args):
        """
        Create GIF to be shown in notebook that can be downloaded
        """
        val = self.dropdown_resolution.value
        # Adapt shown resolution to keep dimensions ratio
        if val == "480p":
            width, height = 400, 300
        elif val == "720p" or val == "1080p" or val == "1440p" or val == "2K":
            width, height = 400, 225
        elif val == "500x500p" or val == "1000x1000p":
            width, height = 400, 400

        # Create visualisable GIF
        with open(self.tmpFileName_movie, "rb") as f:
            gif_bytes = f.read()
            gif = Image(data=gif_bytes, format="gif", width=width, height=height)
        # Add GIF to output view
        self.output_gif.append_display_data(gif)

    def on_orientation_change(self, *args):
        """
        Show real-time camera position
        """
        with self.output_camera_position:
            self.output_camera_position.clear_output()
            position = [round(x, 1) for x in self.view._camera_orientation]
            print(position)

    def change_camera_position(self, *args):
        """
        Update camera position from array
        """
        orientation = json.loads(self.text_orientation.value)
        with self.output_camera_position_error:
            self.output_camera_position_error.clear_output()
            if type(orientation) is not list or len(orientation) != 16:
                print("Orientation must be a length 16 list")
            else:
                self.view._set_camera_orientation(orientation)

    def set_camera(self, *args, direction="x"):
        # See here for rotation matrix https://www.brainvoyager.com/bv/doc/UsersGuide/CoordsAndTransforms/SpatialTransformationMatrices.html
        """
        Set camera orientation along x,y,z axis
        """
        theta = np.pi / 2
        Rx = np.array(
            [
                [self.zoom, 0, 0, 0],
                [0, self.zoom * np.cos(theta), self.zoom * np.sin(theta), 0],
                [0, -self.zoom * np.sin(theta), self.zoom * np.cos(theta), 0],
                [0, 0, 0, 1],
            ]
        )
        Ry = np.array(
            [
                [self.zoom * np.cos(theta), 0, -self.zoom * np.sin(theta), 0],
                [0, self.zoom, 0, 0],
                [self.zoom * np.sin(theta), 0, self.zoom * np.cos(theta), 0],
                [0, 0, 0, 1],
            ]
        )
        Rz = np.array(
            [
                [self.zoom * np.cos(theta), -self.zoom * np.sin(theta), 0, 0],
                [self.zoom * np.sin(theta), self.zoom * np.cos(theta), 0, 0],
                [0, 0, self.zoom, 0],
                [0, 0, 0, 1],
            ]
        )
        if direction == "x":
            self.view._set_camera_orientation([x for x in Rx.flatten()])
        if direction == "y":
            self.view._set_camera_orientation([x for x in Ry.flatten()])
        if direction == "z":
            self.view._set_camera_orientation([x for x in Rz.flatten()])