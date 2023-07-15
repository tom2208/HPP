import numpy as np
import pylab as plt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import hpp

right = 0
down = 1
left = 2
up = 3

width = 200
height = 300
lattice = np.zeros((width, height))

x = int(width / 2) - 50
y = int(height / 2) - 50
lattice[x:x+101, y:y+101] = 15

iterations = 10000


def is_particle(x, y, direction):
    return int(lattice[x][y] / (2 ** direction)) % 2


def is_not_particle(x, y, direction):
    return 1 - is_particle(x, y, direction)


def move_x(x, move):
    return (x + move) % width


def move_y(y, move):
    return (y + move) % height


def psi(x, y):
    horizontal_crossing = is_particle(move_x(x, -1), y, right) * is_particle(move_x(x, 1), y, left)
    vertical_abstinence = is_not_particle(x, move_y(y, -1), down) * is_not_particle(x, move_y(y, 1), up)
    just_horizontal_collision = horizontal_crossing * vertical_abstinence

    horizontal_abstinence = is_not_particle(move_x(x, -1), y, right) * is_not_particle(move_x(x, 1), y, left)
    vertical_moving_out = is_particle(x, move_y(y, -1), down) * is_particle(x, move_y(y, 1), up)
    just_vertical_moving_out = horizontal_abstinence * vertical_moving_out

    return just_horizontal_collision - just_vertical_moving_out


def is_particle_in_next_iteration(x, y, direction):
    if direction == right:
        return is_particle(move_x(x, - 1), y, right) - psi(x, y)
    elif direction == down:
        return is_particle(x, move_y(y, -1), down) + psi(x, y)
    elif direction == left:
        return is_particle(move_x(x, 1), y, left) - psi(x, y)
    else:  # direction == up
        return is_particle(x, move_y(y, 1), up) + psi(x, y)


def calculate_particle_position(x, y):
    particle_sum = 0
    for direction in range(0, 4):
        particle_sum += int(is_particle_in_next_iteration(x, y, direction) * (2 ** direction))
    return particle_sum


def new_particle_positions():
    new_lattice = np.zeros((width, height))

    with Pool(cpu_count()) as pool:
        positions = [(x, y) for x in range(width) for y in range(height)]
        results = pool.starmap(calculate_particle_position, positions)

        for (x, y), particle_sum in zip(positions, results):
            new_lattice[x][y] = particle_sum

    return new_lattice


def hamming_weight(particle):
    new_particle = (int(particle) & 0x55555555) + (int(particle) >> 1 & 0x55555555)
    new_particle = (new_particle & 0x33333333) + (new_particle >> 2 & 0x33333333)
    return new_particle


def convert_lattice_to_hamming():
    for x in range(width):
        for y in range(height):
            lattice[x][y] = hamming_weight(lattice[x][y])


def main_loop():
    global lattice

    for i in tqdm(range(iterations)):
        new_lattice = new_particle_positions()
        if (i % 10) == 0:
            convert_lattice_to_hamming()
            plt.pcolormesh(lattice, cmap='gray')
            plt.axis('equal')
            plt.axis('off')
            plt.savefig('images/' + str(i).zfill(4) + '.png', bbox_inches='tight')
            plt.clf()
        # plt.show()
        lattice = new_lattice


if __name__ == '__main__':
    main_loop()
