from threading import Event, Thread

import cv2
import numpy as np
import torch
from einops import rearrange
from ipycanvas import Canvas, hold_canvas  # noqa
from ipywidgets import Button, HBox, VBox

from controllable_nca.utils import create_2d_circular_mask, rgb


def to_numpy_rgb(x, use_rgb=False):
    return rearrange(
        np.squeeze(rgb(x, use_rgb).detach().cpu().numpy()), "c x y -> x y c"
    )


class ControllableNCAImageVisualizer:
    def __init__(
        self,
        trainer,
        image_size,
        rgb: bool = False,
        canvas_scale=5,
        damage_radius: int = 5,
    ):
        self.trainer = trainer
        self.current_state = None
        self.current_embedding = None

        self.image_size = image_size
        self.rgb = rgb
        self.canvas_scale = canvas_scale
        self.canvas_size = self.image_size * self.canvas_scale

        self.canvas = Canvas(width=self.canvas_size, height=self.canvas_size)
        self.canvas.on_mouse_down(self.handle_mouse_down)
        self.stopped = Event()

        x = self.trainer.target_dataset.x
        d = self.trainer.target_dataset.digits.to(trainer.device)
        with torch.no_grad():
            self.embeddings = (
                self.trainer.nca.image_encoder(x)
                .view(x.size(0), self.trainer.nca.num_channels, 1, 1)
                .repeat(1, 1, x.size(-1), x.size(-1))
            )
            self.digit_embeddings = (
                self.trainer.nca.image_encoder(d)
                .view(x.size(0), self.trainer.nca.num_channels, 1, 1)
                .repeat(1, 1, x.size(-1), x.size(-1))
            )
            self.embeddings = torch.cat([self.embeddings, self.digit_embeddings], dim=0)
        print(self.embeddings.size())
        self.current_embedding = self.embeddings[0:1]

        self.device = self.trainer.device
        self.damage_radius = damage_radius
        self.current_state = self.trainer.nca.generate_seed(1).to(self.device)

        def button_fn(class_num):
            def start(btn):
                self.current_embedding = self.embeddings[class_num : class_num + 1]
                if self.stopped.isSet():
                    self.stopped.clear()
                    Thread(target=self.loop).start()

            return start

        button_list = []
        for i in range(len(self.trainer.target_dataset.EMOJI)):
            button_list.append(Button(description=self.trainer.target_dataset.EMOJI[i]))
            button_list[-1].on_click(button_fn(i))

        self.vbox = VBox(button_list)

        self.stop_btn = Button(description="Stop")

        def stop(btn):
            if not self.stopped.isSet():
                self.stopped.set()

        self.stop_btn.on_click(stop)

    def handle_mouse_down(self, xpos, ypos):
        in_x = int(xpos / self.canvas_scale)
        in_y = int(ypos / self.canvas_scale)

        mask = create_2d_circular_mask(
            self.image_size,
            self.image_size,
            (in_x, in_y),
            radius=self.damage_radius,
        )
        self.current_state[0][:, mask] *= 0.0

    def draw_image(self, rgb):
        with hold_canvas(self.canvas):
            rgb = np.squeeze(rearrange(rgb, "b c w h -> b w h c"))
            self.canvas.clear()  # Clear the old animation step
            self.canvas.put_image_data(
                cv2.resize(
                    rgb * 255.0,
                    (self.canvas_size, self.canvas_size),
                    interpolation=cv2.INTER_NEAREST,
                ),
                0,
                0,
            )

    def loop(self):
        with torch.no_grad():
            self.current_state = self.trainer.nca.generate_seed(1).to(self.device)
            while not self.stopped.wait(0.02):  # the first call is in `interval` secs
                # update_particle_locations()
                self.draw_image(self.trainer.to_rgb(self.current_state))
                self.current_state, self.current_embedding = self.trainer.nca(
                    (self.current_state, self.current_embedding)
                )

    def visualize(self):
        Thread(target=self.loop).start()
        display(self.canvas, HBox([self.stop_btn, self.vbox]))  # noqa
