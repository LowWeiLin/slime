import numpy as np
import pygame as pg

import scipy.stats as st

import cupy as cp
from cupyx.scipy.ndimage import convolve


def gkern(kernlen=21, nsig=20):
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d / kern2d.sum()


class SlimeWorld:

    WIDTH = 800
    HEIGHT = 800
    cells = cp.zeros((WIDTH, HEIGHT, 3), dtype=cp.float16)

    trail_color = cp.array([255, 255, 255])
    trail_reduction_factor = 0.99
    trail_gaussian_filer_sigma = 0.4
    trail_kernel = cp.array(gkern())
    trail_kernel = trail_kernel[:, :, None]

    sense_dist = 5
    sense_angles = [np.deg2rad(-30), 0, np.deg2rad(30)]
    turn_angle = np.deg2rad(10)
    turn_angle_random = np.deg2rad(10)

    num_slimes = 20000
    slime_pos = cp.random.random((num_slimes, 2)) * cp.array([WIDTH, HEIGHT])
    slime_angle = cp.random.random(num_slimes) * 2 * np.pi

    def initialize():
        pass

    def tick():
        SlimeWorld.move_slimes()
        SlimeWorld.leave_slime_trail()
        SlimeWorld.diffuse_slime_trail()
        SlimeWorld.sense_and_turn()

    def clip(pos):
        cp.clip(
            pos[:, 0],
            0,
            SlimeWorld.WIDTH - 1,
            out=pos[:, 0],
        )
        cp.clip(
            pos[:, 1],
            0,
            SlimeWorld.HEIGHT - 1,
            out=pos[:, 1],
        )

    def sense_at_angle(angle, channel=0):
        sense_angle = SlimeWorld.slime_angle + angle
        sense_dir = (
            cp.array([cp.sin(sense_angle), cp.cos(sense_angle)]).T
            * SlimeWorld.sense_dist
        )
        sense_pos = SlimeWorld.slime_pos + sense_dir
        SlimeWorld.clip(sense_pos)
        int_sense_pos = sense_pos.astype(int)
        return SlimeWorld.cells[int_sense_pos[:, 0], int_sense_pos[:, 1], channel]

    def sense_and_turn():
        sensors = cp.array(
            [SlimeWorld.sense_at_angle(angle) for angle in SlimeWorld.sense_angles]
        )
        sensors_argmax = cp.argmax(sensors, axis=0)
        random_angle = (
            cp.random.random(SlimeWorld.num_slimes) * 2 * SlimeWorld.turn_angle_random
        ) - SlimeWorld.turn_angle_random

        SlimeWorld.slime_angle[sensors_argmax == 1] -= SlimeWorld.turn_angle
        SlimeWorld.slime_angle[sensors_argmax == 2] += SlimeWorld.turn_angle
        SlimeWorld.slime_angle += random_angle

    def move_slimes():
        slime_dir = cp.array(
            [cp.sin(SlimeWorld.slime_angle), cp.cos(SlimeWorld.slime_angle)]
        ).T
        SlimeWorld.slime_pos += slime_dir

        # Change direction if out of bounds
        xoob = (SlimeWorld.slime_pos[:, 0] < 0) | (
            SlimeWorld.slime_pos[:, 0] > SlimeWorld.WIDTH - 1
        )
        yoob = (SlimeWorld.slime_pos[:, 1] < 0) | (
            SlimeWorld.slime_pos[:, 1] > SlimeWorld.HEIGHT - 1
        )

        # Bounce off walls
        if cp.any(xoob):
            slime_dir[xoob, 0] *= -1
        if cp.any(yoob):
            slime_dir[yoob, 1] *= -1
        if cp.any(xoob | yoob):
            SlimeWorld.slime_angle = cp.arctan2(slime_dir[:, 0], slime_dir[:, 1])
            # Clip if out of bounds
            SlimeWorld.clip(SlimeWorld.slime_pos)

    def leave_slime_trail():
        int_pos = SlimeWorld.slime_pos.astype(int)
        SlimeWorld.cells[int_pos[:, 0], int_pos[:, 1]] = SlimeWorld.trail_color

    def diffuse_slime_trail():
        cp.multiply(
            SlimeWorld.cells,
            SlimeWorld.trail_reduction_factor,
            out=SlimeWorld.cells,
            casting="unsafe",
        )
        convolve(SlimeWorld.cells, SlimeWorld.trail_kernel, output=SlimeWorld.cells)


class Renderer:

    cellsize = 1
    WIDTH = SlimeWorld.cells.shape[0] * cellsize
    HEIGHT = SlimeWorld.cells.shape[1] * cellsize

    def initialize():

        # initialize pygame
        pg.init()
        Renderer.screen = pg.display.set_mode((Renderer.WIDTH, Renderer.HEIGHT))
        Renderer.clock = pg.time.Clock()
        Renderer.create_fonts([32, 16, 14, 8])

        # create a surface with the size as the array
        Renderer.surface = pg.Surface(
            (SlimeWorld.cells.shape[0], SlimeWorld.cells.shape[1])
        )

    def create_fonts(font_sizes_list):
        "Creates different fonts with one list"
        Renderer.fonts = []
        for size in font_sizes_list:
            Renderer.fonts.append(pg.font.SysFont("Arial", size))

    def render(fnt, what, color, where):
        "Renders the fonts as passed from display_fps"
        text_to_show = fnt.render(what, 0, pg.Color(color))
        Renderer.screen.blit(text_to_show, where)

    def display_fps():
        "Data that will be rendered and blitted in _display"
        Renderer.render(
            Renderer.fonts[0],
            what=str(int(Renderer.clock.get_fps())),
            color="white",
            where=(0, 0),
        )

    def render_loop():

        # game loop
        running = True
        while running:
            Renderer.clock.tick(60)

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False

            SlimeWorld.tick()

            array_buffer = cp.asnumpy(SlimeWorld.cells).astype(np.uint8)
            pg.surfarray.blit_array(Renderer.surface, array_buffer)
            scaled_surface = pg.transform.scale(
                Renderer.surface, (Renderer.WIDTH, Renderer.HEIGHT)
            )

            Renderer.screen.fill((0, 0, 0))
            # blit the transformed surface onto the screen
            Renderer.screen.blit(scaled_surface, (0, 0))
            Renderer.display_fps()

            pg.display.update()

        pg.quit()


if __name__ == "__main__":

    SlimeWorld.initialize()
    Renderer.initialize()
    Renderer.render_loop()