# %%
"""
This code is reorganized to assure beeing isolated from all the other .py in the project.
It is a complete version of the ant colony optimization algorithm.
It is a sequential version of the code, used for comparison with the parallelized version.
"""
import numpy as np
import direction as d
import pygame as pg
import pandas as pd

# deactivating the numpy paralelism
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

plot = False

UNLOADED, LOADED = False, True

NORTH = 1
EAST  = 2
SOUTH = 4
WEST  = 8

exploration_coefs = 0.

class Colony:
    """
    Represent an ant colony. Ants are not individualized for performance reasons!

    Inputs :
        nb_ants  : Number of ants in the anthill
        pos_init : Initial positions of ants (anthill position)
        max_life : Maximum life that ants can reach
    """
    def __init__(self, nb_ants, pos_init, max_life):
        # Each ant has is own unique random seed
        self.seeds = np.arange(1, nb_ants+1, dtype=np.int64)
        # State of each ant : loaded or unloaded
        self.is_loaded = np.zeros(nb_ants, dtype=np.int8)
        # Compute the maximal life amount for each ant :
        #   Updating the random seed :
        self.seeds[:] = np.mod(16807*self.seeds[:], 2147483647)
        # Amount of life for each ant = 75% à 100% of maximal ants life
        self.max_life = max_life * np.ones(nb_ants, dtype=np.int32)
        self.max_life -= np.int32(max_life*(self.seeds/2147483647.))//4
        # Ages of ants : zero at beginning
        self.age = np.zeros(nb_ants, dtype=np.int64)
        # History of the path taken by each ant. The position at the ant's age represents its current position.
        self.historic_path = np.zeros((nb_ants, max_life+1, 2), dtype=np.int16)
        self.historic_path[:, 0, 0] = pos_init[0]
        self.historic_path[:, 0, 1] = pos_init[1]
        # Direction in which the ant is currently facing (depends on the direction it came from).
        self.directions = d.DIR_NONE*np.ones(nb_ants, dtype=np.int8)
        self.sprites = []
        img = pg.image.load("ants.png").convert_alpha()
        for i in range(0, 32, 8):
            self.sprites.append(pg.Surface.subsurface(img, i, 0, 8, 8))

    def return_to_nest(self, loaded_ants, pos_nest, food_counter):
        """
        Function that returns the ants carrying food to their nests.

        Inputs :
            loaded_ants: Indices of ants carrying food
            pos_nest: Position of the nest where ants should go
            food_counter: Current quantity of food in the nest

        Returns the new quantity of food
        """
        self.age[loaded_ants] -= 1

        in_nest_tmp = self.historic_path[loaded_ants, self.age[loaded_ants], :] == pos_nest
        if in_nest_tmp.any():
            in_nest_loc = np.nonzero(np.logical_and(in_nest_tmp[:, 0], in_nest_tmp[:, 1]))[0]
            if in_nest_loc.shape[0] > 0:
                in_nest = loaded_ants[in_nest_loc]
                self.is_loaded[in_nest] = UNLOADED
                self.age[in_nest] = 0
                food_counter += in_nest_loc.shape[0]
        return food_counter

    def explore(self, unloaded_ants, the_maze, pos_food, pos_nest, pheromones):
        """
        Management of unloaded ants exploring the maze.

        Inputs:
            unloadedAnts: Indices of ants that are not loaded
            maze        : The maze in which ants move
            posFood     : Position of food in the maze
            posNest     : Position of the ants' nest in the maze
            pheromones  : The pheromone map (which also has ghost cells for
                          easier edge management)

        Outputs: None
        """
        # Update of the random seed (for manual pseudo-random) applied to all unloaded ants
        self.seeds[unloaded_ants] = np.mod(16807*self.seeds[unloaded_ants], 2147483647)

        # Calculating possible exits for each ant in the maze:
        old_pos_ants = self.historic_path[range(0, self.seeds.shape[0]), self.age[:], :]
        has_north_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], NORTH) > 0
        has_east_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], EAST) > 0
        has_south_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], SOUTH) > 0
        has_west_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], WEST) > 0

        # Reading neighboring pheromones:
        north_pos = np.copy(old_pos_ants)
        north_pos[:, 1] += 1
        north_pheromone = pheromones.pheromon[north_pos[:, 0], north_pos[:, 1]]*has_north_exit

        east_pos = np.copy(old_pos_ants)
        east_pos[:, 0] += 1
        east_pos[:, 1] += 2
        east_pheromone = pheromones.pheromon[east_pos[:, 0], east_pos[:, 1]]*has_east_exit

        south_pos = np.copy(old_pos_ants)
        south_pos[:, 0] += 2
        south_pos[:, 1] += 1
        south_pheromone = pheromones.pheromon[south_pos[:, 0], south_pos[:, 1]]*has_south_exit

        west_pos = np.copy(old_pos_ants)
        west_pos[:, 0] += 1
        west_pheromone = pheromones.pheromon[west_pos[:, 0], west_pos[:, 1]]*has_west_exit

        max_pheromones = np.maximum(north_pheromone, east_pheromone)
        max_pheromones = np.maximum(max_pheromones, south_pheromone)
        max_pheromones = np.maximum(max_pheromones, west_pheromone)

        # Calculating choices for all ants not carrying food (for others, we calculate but it doesn't matter)
        choices = self.seeds[:] / 2147483647.

        # Ants explore the maze by choice or if no pheromone can guide them:
        ind_exploring_ants = np.nonzero(
            np.logical_or(choices[unloaded_ants] <= exploration_coefs, max_pheromones[unloaded_ants] == 0.))[0]
        if ind_exploring_ants.shape[0] > 0:
            ind_exploring_ants = unloaded_ants[ind_exploring_ants]
            valid_moves = np.zeros(choices.shape[0], np.int8)
            nb_exits = has_north_exit * np.ones(has_north_exit.shape) + has_east_exit * np.ones(has_east_exit.shape) + \
                has_south_exit * np.ones(has_south_exit.shape) + has_west_exit * np.ones(has_west_exit.shape)
            while np.any(valid_moves[ind_exploring_ants] == 0):
                # Calculating indices of ants whose last move was not valid:
                ind_ants_to_move = ind_exploring_ants[valid_moves[ind_exploring_ants] == 0]
                self.seeds[:] = np.mod(16807*self.seeds[:], 2147483647)
                # Choosing a random direction:
                dir = np.mod(self.seeds[ind_ants_to_move], 4)
                old_pos = self.historic_path[ind_ants_to_move, self.age[ind_ants_to_move], :]
                new_pos = np.copy(old_pos)
                new_pos[:, 1] -= np.logical_and(dir == d.DIR_WEST,
                                                has_west_exit[ind_ants_to_move]) * np.ones(new_pos.shape[0], dtype=np.int16)
                new_pos[:, 1] += np.logical_and(dir == d.DIR_EAST,
                                                has_east_exit[ind_ants_to_move]) * np.ones(new_pos.shape[0], dtype=np.int16)
                new_pos[:, 0] -= np.logical_and(dir == d.DIR_NORTH,
                                                has_north_exit[ind_ants_to_move]) * np.ones(new_pos.shape[0], dtype=np.int16)
                new_pos[:, 0] += np.logical_and(dir == d.DIR_SOUTH,
                                                has_south_exit[ind_ants_to_move]) * np.ones(new_pos.shape[0], dtype=np.int16)
                # Valid move if we didn't stay in place due to a wall
                valid_moves[ind_ants_to_move] = np.logical_or(new_pos[:, 0] != old_pos[:, 0], new_pos[:, 1] != old_pos[:, 1])
                # and if we're not in the opposite direction of the previous move (and if there are other exits)
                valid_moves[ind_ants_to_move] = np.logical_and(
                    valid_moves[ind_ants_to_move],
                    np.logical_or(dir != 3-self.directions[ind_ants_to_move], nb_exits[ind_ants_to_move] == 1))
                # Calculating indices of ants whose move we just validated:
                ind_valid_moves = ind_ants_to_move[np.nonzero(valid_moves[ind_ants_to_move])[0]]
                # For these ants, we update their positions and directions
                self.historic_path[ind_valid_moves, self.age[ind_valid_moves] + 1, :] = new_pos[valid_moves[ind_ants_to_move] == 1, :]
                self.directions[ind_valid_moves] = dir[valid_moves[ind_ants_to_move] == 1]

        ind_following_ants = np.nonzero(np.logical_and(choices[unloaded_ants] > exploration_coefs,
                                                       max_pheromones[unloaded_ants] > 0.))[0]
        if ind_following_ants.shape[0] > 0:
            ind_following_ants = unloaded_ants[ind_following_ants]
            self.historic_path[ind_following_ants, self.age[ind_following_ants] + 1, :] = \
                self.historic_path[ind_following_ants, self.age[ind_following_ants], :]
            max_east = (east_pheromone[ind_following_ants] == max_pheromones[ind_following_ants])
            self.historic_path[ind_following_ants, self.age[ind_following_ants]+1, 1] += \
                max_east * np.ones(ind_following_ants.shape[0], dtype=np.int16)
            max_west = (west_pheromone[ind_following_ants] == max_pheromones[ind_following_ants])
            self.historic_path[ind_following_ants, self.age[ind_following_ants]+1, 1] -= \
                max_west * np.ones(ind_following_ants.shape[0], dtype=np.int16)
            max_north = (north_pheromone[ind_following_ants] == max_pheromones[ind_following_ants])
            self.historic_path[ind_following_ants, self.age[ind_following_ants]+1, 0] -= max_north * np.ones(ind_following_ants.shape[0], dtype=np.int16)
            max_south = (south_pheromone[ind_following_ants] == max_pheromones[ind_following_ants])
            self.historic_path[ind_following_ants, self.age[ind_following_ants]+1, 0] += max_south * np.ones(ind_following_ants.shape[0], dtype=np.int16)

        # Aging one unit for the age of ants not carrying food
        if unloaded_ants.shape[0] > 0:
            self.age[unloaded_ants] += 1

        # Killing ants at the end of their life:
        ind_dying_ants = np.nonzero(self.age == self.max_life)[0]
        if ind_dying_ants.shape[0] > 0:
            self.age[ind_dying_ants] = 0
            self.historic_path[ind_dying_ants, 0, 0] = pos_nest[0]
            self.historic_path[ind_dying_ants, 0, 1] = pos_nest[1]
            self.directions[ind_dying_ants] = d.DIR_NONE

        # For ants reaching food, we update their states:
        ants_at_food_loc = np.nonzero(np.logical_and(self.historic_path[unloaded_ants, self.age[unloaded_ants], 0] == pos_food[0],
                                                     self.historic_path[unloaded_ants, self.age[unloaded_ants], 1] == pos_food[1]))[0]
        if ants_at_food_loc.shape[0] > 0:
            ants_at_food = unloaded_ants[ants_at_food_loc]
            self.is_loaded[ants_at_food] = True

    def advance(self, the_maze, pos_food, pos_nest, pheromones, food_counter=0):
        loaded_ants = np.nonzero(self.is_loaded == True)[0]
        unloaded_ants = np.nonzero(self.is_loaded == False)[0]
        if loaded_ants.shape[0] > 0:
            food_counter = self.return_to_nest(loaded_ants, pos_nest, food_counter)
        if unloaded_ants.shape[0] > 0:
            self.explore(unloaded_ants, the_maze, pos_food, pos_nest, pheromones)

        old_pos_ants = self.historic_path[range(0, self.seeds.shape[0]), self.age[:], :]
        has_north_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], NORTH) > 0
        has_east_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], EAST) > 0
        has_south_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], SOUTH) > 0
        has_west_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], WEST) > 0
        # Marking pheromones:
        [pheromones.mark(self.historic_path[i, self.age[i], :],
                         [has_north_exit[i], has_east_exit[i], has_west_exit[i], has_south_exit[i]]) for i in range(self.directions.shape[0])]
        return food_counter
    
    def display(self, screen):
        [screen.blit(self.sprites[self.directions[i]], (8*self.historic_path[i, self.age[i], 1], 8*self.historic_path[i, self.age[i], 0])) for i in range(self.directions.shape[0])]

    def get_ants_data_for_mpi(self):
        # Initialize an empty list to store the ants' data
        ants_data = []

        # Iterate over the ants
        for ant in self.ants:
            # Get the ant's position and rotation
            position = ant.position
            rotation = ant.rotation

            # Append the position and rotation to the list
            ants_data.append((position, rotation))

        # Return the list of ants' data
        return ants_data

class Maze:
    """
    Builds a maze of given dimensions by building the NumPy array maze describing the maze.

    Inputs:
        dimensions: Tuple containing two integers describing the height and length of the maze.
        seed: The random seed used to generate the maze. The same seed produces the same maze.
    """
    def __init__(self, dimensions, seed):

        self.cases_img = []
        self.maze  = np.zeros(dimensions, dtype=np.int8)
        is_visited = np.zeros(dimensions, dtype=np.int8)
        historic = []

        # We choose the central cell as the initial cell.
        cur_ind = (dimensions[0]//2, dimensions[1]//2)
        historic.append(cur_ind)
        while (len(historic) > 0):
            cur_ind = historic[-1]
            is_visited[cur_ind] = 1
            # First, we check if there is at least one unvisited neighboring cell of the current cell:
            #   1. Calculating the neighbors of the current cell:
            neighbours         = []
            neighbours_visited = []
            direction          = []
            if cur_ind[1] > 0 and is_visited[cur_ind[0], cur_ind[1]-1] == 0:  # West cell no visited
                neighbours.append((cur_ind[0], cur_ind[1]-1))
                direction.append((WEST, EAST))
            if cur_ind[1] < dimensions[1]-1 and is_visited[cur_ind[0], cur_ind[1]+1] == 0:  # East cell
                neighbours.append((cur_ind[0], cur_ind[1]+1))
                direction.append((EAST, WEST))
            if cur_ind[0] < dimensions[0]-1 and is_visited[cur_ind[0]+1, cur_ind[1]] == 0:  # South cell
                neighbours.append((cur_ind[0]+1, cur_ind[1]))
                direction.append((SOUTH, NORTH))
            if cur_ind[0] > 0 and is_visited[cur_ind[0]-1, cur_ind[1]] == 0:  # North cell
                neighbours.append((cur_ind[0]-1, cur_ind[1]))
                direction.append((NORTH, SOUTH))
            if len(neighbours) > 0:  # In this case, a cell is non visited
                neighbours = np.array(neighbours)
                direction  = np.array(direction)
                seed = (16807*seed) % 2147483647
                chosen_dir = seed % len(neighbours)
                dir        = direction[chosen_dir]
                historic.append((neighbours[chosen_dir, 0], neighbours[chosen_dir, 1]))
                self.maze[cur_ind] |= dir[0]
                self.maze[neighbours[chosen_dir, 0], neighbours[chosen_dir, 1]] |= dir[1]
                is_visited[cur_ind] = 1
            else:
                historic.pop()
        #  Load patterns for maze display :
        img = pg.image.load("cases.png").convert_alpha()
        for i in range(0, 128, 8):
            self.cases_img.append(pg.Surface.subsurface(img, i, 0, 8, 8))

    def display(self):
        """
        Create a picture of the maze :
        """
        maze_img = pg.Surface((8*self.maze.shape[1], 8*self.maze.shape[0]), flags=pg.SRCALPHA)
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                maze_img.blit(self.cases_img[self.maze[i, j]], (j*8, i*8))

        return maze_img

class Pheromon:
    """
    """
    def __init__(self, the_dimensions, the_food_position, the_alpha=0.7, the_beta=0.9999):
        self.alpha = the_alpha
        self.beta  = the_beta
        #  We add a row of cells at the bottom, top, left, and right to facilitate edge management in vectorized form
        self.pheromon = np.zeros((the_dimensions[0]+2, the_dimensions[1]+2), dtype=np.double)
        self.pheromon[the_food_position[0]+1, the_food_position[1]+1] = 1.
        self.total_pheromones = []

    def do_evaporation(self, the_pos_food):
        self.pheromon = self.beta * self.pheromon
        self.pheromon[the_pos_food[0]+1, the_pos_food[1]+1] = 1.

    def mark(self, the_position, has_WESN_exits):
        assert(the_position[0] >= 0)
        assert(the_position[1] >= 0)
        cells = np.array([self.pheromon[the_position[0]+1, the_position[1]] if has_WESN_exits[d.DIR_WEST] else 0.,
                          self.pheromon[the_position[0]+1, the_position[1]+2] if has_WESN_exits[d.DIR_EAST] else 0.,
                          self.pheromon[the_position[0]+2, the_position[1]+1] if has_WESN_exits[d.DIR_SOUTH] else 0.,
                          self.pheromon[the_position[0], the_position[1]+1] if has_WESN_exits[d.DIR_NORTH] else 0.], dtype=np.double)
        pheromones = np.maximum(cells, 0.)
        self.pheromon[the_position[0]+1, the_position[1]+1] = self.alpha*np.max(pheromones) + (1-self.alpha)*0.25*pheromones.sum()

    def getColor(self, i: int, j: int):
        val = max(min(self.pheromon[i, j], 1), 0)
        return [255*(val > 1.E-16), 255*val, 128.]

    def display(self, screen):
        [[screen.fill(self.getColor(i, j), (8*(j-1), 8*(i-1), 8, 8)) for j in range(1, self.pheromon.shape[1]-1)] for i in range(1, self.pheromon.shape[0]-1)]

# %%

if __name__ == "__main__":
    import sys
    import time
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    pg.init()
    size_laby = 25, 25
    if len(sys.argv) > 2:
        size_laby = int(sys.argv[1]),int(sys.argv[2])

    resolution = size_laby[1]*8, size_laby[0]*8
    screen = pg.display.set_mode(resolution)
    nb_ants = size_laby[0]*size_laby[1]//4
    max_life = 500
    if len(sys.argv) > 3:
        max_life = int(sys.argv[3])
    pos_food = size_laby[0]-1, size_laby[1]-1
    pos_nest = 0, 0
    a_maze = Maze(size_laby, 12345)
    ants = Colony(nb_ants, pos_nest, max_life)
    unloaded_ants = np.array(range(nb_ants))
    alpha = 0.9
    beta  = 0.99
    if len(sys.argv) > 4:
        alpha = float(sys.argv[4])
    if len(sys.argv) > 5:
        beta = float(sys.argv[5])
    pherom = Pheromon(size_laby, pos_food, alpha, beta)
    mazeImg = a_maze.display()
    food_counter = 0
    
    # for evaluation purposes
    total_pheromones_i = []
    total_fps_i = []
    total_food_i = []
    total_time_per_iteration_i = []
    flag_100 = 0
    flag_500 = 0
    i_counter = 0
    
    snapshop_taken = False
    while True:
        
        if i_counter == 2000:
            break
        
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                exit(0)

        deb = time.time()
        pherom.display(screen)
        screen.blit(mazeImg, (0, 0))
        ants.display(screen)
        pg.display.update()
                
        food_counter = ants.advance(a_maze, pos_food, pos_nest, pherom, food_counter)
        pherom.do_evaporation(pos_food)
        end = time.time()
        
        total_pheromones_i.append(pherom.pheromon.sum())
        total_fps_i.append(1./(end-deb))
        total_time_per_iteration_i.append(end-deb)
        total_food_i.append(food_counter)
        
        if food_counter == 1 and not snapshop_taken:
            pg.image.save(screen, "MyFirstFood.png")
            snapshop_taken = True
        
        if food_counter > 100 and flag_100 == 0:
            print("100 food reached")
            flag_100 = 1
            
        if food_counter > 500 and flag_500 == 0:
            print("500 food reached")
            flag_500 = 1
            
        i_counter += 1
        
        
        
        # pg.time.wait(500)
        # print(f"FPS : {1./(end-deb):6.2f}, nourriture : {food_counter:7d}", end='\r')
        
    average_fps = sum(total_fps_i) / len(total_fps_i)
    max_fps = max(total_fps_i)
    min_fps = min(total_fps_i)
    std_dev_fps = np.std(total_fps_i)
    
    average_time_per_iteration = sum(total_time_per_iteration_i) / len(total_time_per_iteration_i)
    total_time = sum(total_time_per_iteration_i)
    max_time = max(total_time_per_iteration_i)
    min_time = min(total_time_per_iteration_i)
    std_dev_time = np.std(total_time_per_iteration_i)
    
    pg.quit()
    
    # %% 
    print(f"Average Time per Iteration: {average_time_per_iteration}")
    print(f"Total time: {total_time}")
    print(f"Max Time per Iteration: {max_time}")
    print(f"Min Time per Iteration: {min_time}")
    print(f"Standard Deviation of Time per Iteration: {std_dev_time}")

    if plot:
        # Calculate the derivative
        food_derivative = np.gradient(total_food_i)

        # %%
        # Convert the list to a numpy array
        food_derivative_np = np.array(food_derivative)
        
        # Set a Seaborn style
        sns.set(style="whitegrid", palette="pastel", font_scale=1.2)

        # Create a figure to hold the subplots, adjust to 3 rows
        fig, axs = plt.subplots(3, 1, figsize=(10, 18))  # Now 3 rows, 1 column

        # First subplot for Total Pheromones
        axs[0].plot(total_pheromones_i, linewidth=1, color='deepskyblue', linestyle='-', label='Total Pheromones')
        # axs[0].set_xlabel('Iteration', fontsize=14)
        axs[0].set_ylabel('Total Pheromones', fontsize=14)
        axs[0].set_title('Evaluation of The Sequencial Program', fontsize=16)
        axs[0].legend(frameon=True, loc='upper left')
        axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)

        # Second subplot for time per iteration
        axs[1].plot(total_time_per_iteration_i, linewidth=1, color='lightcoral', linestyle='-', label='Time per Iteration')
        # axs[1].set_xlabel('Iteration', fontsize=14)
        axs[1].set_ylabel('Time per iteration', fontsize=14)
        # axs[1].set_title('Time per Iteration Over Time', fontsize=16)
        axs[1].axhline(y=average_time_per_iteration, color='green', linestyle='--', linewidth=1, label='Average FPS')
        axs[1].legend(frameon=True, loc='upper left')
        axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)


        # Third subplot for Total Food Collected
        axs[2].plot(food_derivative[:-1], linewidth=1, color='orange', linestyle='-', label='Food Derivative')
        axs[2].set_xlabel('Iteration', fontsize=14)
        axs[2].set_ylabel('Food Units', fontsize=14)
        axs[2].legend(frameon=True, loc='upper left')
        axs[2].grid(True, which='both', linestyle='--', linewidth=0.5)

        # Adjust layout
        plt.tight_layout(pad=3.0)

        # Save the figure
        plt.savefig("combined_plots.png")

        # Show the plot
        plt.show()
        
    # %%
    # Create a dictionary with the data
    data = {
    "Iterations": list(range(len(total_pheromones_i))),
    "Total Pheromones": total_pheromones_i,
    "Time per Iteration": total_time_per_iteration_i,
    "Total Food": total_food_i
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Specify the filename
    filename = "simulation_data.xlsx"

    # Save the DataFrame to an Excel file
    df.to_excel(filename, index=False, engine='openpyxl')

    print(f"Data saved to {filename}.")

"""
Last test:
Average FPS: 182.96062959368106
Max FPS: 500.75262655205347
Min FPS: 33.25381748989138
Standard Deviation of FPS: 26.72799187010939
Total time: 14.188844921255443
"""