"""
Module managing an ant colony in a labyrinth.
"""
import numpy as np
import maze
import pheromone
import direction as d
import pygame as pg

UNLOADED, LOADED = False, True

exploration_coefs = 0.

# deactivating the numpy paralelism
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'


class Colony:
    """
    Represent an ant colony. Ants are not individualized for performance reasons!

    Inputs :
        nb_ants  : Number of ants in the anthill
        pos_init : Initial positions of ants (anthill position)
        max_life : Maximum life that ants can reach
    """
    def __init__(self, nb_ants, pos_init, max_life, rank):
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
        
        # prevents the other cores from displaying
        if rank == 0:
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
        has_north_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze.NORTH) > 0
        has_east_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze.EAST) > 0
        has_south_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze.SOUTH) > 0
        has_west_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze.WEST) > 0

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

    def advance(self, the_maze, pos_food, pos_nest, pheromones, phero, food_counter=0):
        loaded_ants = np.nonzero(self.is_loaded == True)[0]
        unloaded_ants = np.nonzero(self.is_loaded == False)[0]
        if loaded_ants.shape[0] > 0:
            food_counter = self.return_to_nest(loaded_ants, pos_nest, food_counter)
        if unloaded_ants.shape[0] > 0:
            self.explore(unloaded_ants, the_maze, pos_food, pos_nest, pheromones)

        old_pos_ants = self.historic_path[range(0, self.seeds.shape[0]), self.age[:], :]
        has_north_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze.NORTH) > 0
        has_east_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze.EAST) > 0
        has_south_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze.SOUTH) > 0
        has_west_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze.WEST) > 0
        # Marking pheromones:
        [pheromones.mark(self.historic_path[i, self.age[i], :],
                         [has_north_exit[i], has_east_exit[i], has_west_exit[i], has_south_exit[i]], phero) for i in range(self.directions.shape[0])]
        return food_counter

    def display(self, screen):
        [screen.blit(self.sprites[self.directions[i]], (8*self.historic_path[i, self.age[i], 1], 8*self.historic_path[i, self.age[i], 0])) for i in range(self.directions.shape[0])]


if __name__ == "__main__":
    import sys
    import time
    from mpi4py import MPI 
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    pg.init()
    size_laby = 25, 25

    resolution = size_laby[1] * 8, size_laby[0] * 8

    nb_ants = size_laby[0] * size_laby[1] // 4
    nb_ants_per_process = nb_ants // (size-1)
    remaining_ants = nb_ants % (size-1)

    # Se houver formigas remanescentes, realoque para os primeiros processos
    if rank == 1:
        nb_ants_per_process += remaining_ants
    
    max_life = 500

    pos_food = size_laby[0] - 1, size_laby[1] - 1
    pos_nest = 0, 0
    alpha = 0.9
    beta  = 0.99

    if rank == 0:
        screen = pg.display.set_mode(resolution)
        ants_global = Colony(nb_ants, pos_nest, max_life, rank)
        unloaded_ants = np.array(range(nb_ants))
    
    ants_local = Colony(nb_ants_per_process, pos_nest, max_life, rank)
    unloaded_ants = np.array(range(nb_ants_per_process))

    a_maze = maze.Maze(size_laby, 12345, rank)
    local_pherom = pheromone.Pheromon(size_laby, pos_food, alpha, beta)

    if rank == 0:
        mazeImg = a_maze.display()

    if len(sys.argv) > 4:
        alpha = float(sys.argv[4])
    if len(sys.argv) > 5:
        beta = float(sys.argv[5])

    food_counter = 0

    total_iterations = 2000  # Número total de iterações
    iteration_count = 0  # Contador de iterações
    total_time = 0.0  # Tempo total das iterações
    
    # for evaluation purposes
    total_pheromones_i = []
    total_fps_i = []
    total_food_i = []
    total_time_per_iteration_i = []
    
    global_pheromon = np.zeros((size_laby[0]+2, size_laby[1]+2), dtype=np.double)

    snapshop_taken = False
    flag = 0
    while iteration_count < total_iterations:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                exit(0)

        if rank != 0:
            # Atualiza o food_counter em cada processo
            comm.Allreduce(local_pherom.pheromon, global_pheromon, op=MPI.MAX)
                
            local_pherom.pheromon[:] = global_pheromon[:]
            
            # local_pherom.setpheromon(global_pheromon)
            
            food_counter = ants_local.advance(a_maze, pos_food, pos_nest, local_pherom, local_pherom.pheromon, food_counter)
            local_pherom.do_evaporation(pos_food)

            # Envia as informações necessárias para o processo 0
            data_to_send = (ants_local.age, ants_local.historic_path, ants_local.directions, food_counter)
            comm.send(data_to_send, dest=0)
            
        else:
            deb = time.time()
            
            food_counter = 0
            
            blank_pheromon = np.zeros((size_laby[0]+2, size_laby[1]+2), dtype=np.double)
            
            # print(local_pherom.pheromon)
            comm.Allreduce(blank_pheromon, global_pheromon, op=MPI.MAX)
            # print(local_pherom.pheromon)
            
            # local_pherom.setpheromon(global_pheromon)
            
            local_pherom.pheromon[:] = global_pheromon[:]
            
            # received_data = [None for _ in range(size)]  # Define the "received_data" variable
            
            age_process = [None for _ in range(size)]
            historic_path_process = [None for _ in range(size)]
            directions_process = [None for _ in range(size)]
            
            global_age = np.zeros(0, dtype=np.int64)
            global_historic_path = np.zeros((0, max_life+1, 2), dtype=np.int16)
            global_directions = np.zeros(0, dtype=np.int64)
            
            for i in range(1, size):
                received_data = comm.recv(source=i)
                age_process[i], historic_path_process[i], directions_process[i], food_counter_local = received_data
                global_directions = np.concatenate((global_directions, directions_process[i]))
                global_age = np.concatenate((global_age, age_process[i]))
                global_historic_path = np.concatenate((global_historic_path, historic_path_process[i]))
                food_counter += food_counter_local
                
            if food_counter > 100 and flag == 0:
                print(f"global n ants: {nb_ants}")
                print(f"Global pheromones: {global_pheromon.shape}")
                print(f"Global age: {global_age.shape}")
                print(f"Global historic path: {global_historic_path.shape}")
                print(f"Global directions: {global_directions.shape}")
                flag = 1
                
                
            ants_global.age = global_age
            ants_global.directions = global_directions
            ants_global.historic_path = global_historic_path
            
            local_pherom.display(screen)
            screen.blit(mazeImg, (0, 0))
            ants_global.display(screen)
            pg.display.update()
            end = time.time()
            
            total_pheromones_i.append(local_pherom.pheromon.sum())
            total_fps_i.append(1./(end-deb))
            total_time_per_iteration_i.append(end-deb)
            total_food_i.append(food_counter)

            fps = 1.0 / (end - deb)

            iteration_time = end - deb
            total_time += iteration_time
            iteration_count += 1

            if food_counter == 1 and not snapshop_taken:
                pg.image.save(screen, "MyFirstFood.png")
                snapshop_taken = True

            print(f"Iteration {iteration_count}: FPS : {fps:6.2f}, Nourriture : {food_counter:7d}, Total Time: {total_time:6.2f} sec", end='\r')
    
    if rank == 0:
        
        average_time_per_iteration = sum(total_time_per_iteration_i) / len(total_time_per_iteration_i)
        total_time = sum(total_time_per_iteration_i)
        max_time = max(total_time_per_iteration_i)
        min_time = min(total_time_per_iteration_i)
        std_dev_time = np.std(total_time_per_iteration_i)
        
        # Print the results
        print(f"Average Time per Iteration: {average_time_per_iteration}")
        print(f"Total time: {total_time}")
        print(f"Max Time per Iteration: {max_time}")
        print(f"Min Time per Iteration: {min_time}")
        print(f"Standard Deviation of Time per Iteration: {std_dev_time}")
        
        # Create a dictionary with the data
        data = {
        # "Iterations": list(range(len(total_pheromones_i))),
        "Total Pheromones": total_pheromones_i,
        # "FPS": total_fps_i,
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
        
        print(f"Average Time per Iteration: {average_time_per_iteration}")
        print(f"Total time: {total_time}")
        print(f"Max Time per Iteration: {max_time}")
        print(f"Min Time per Iteration: {min_time}")
        print(f"Standard Deviation of Time per Iteration: {std_dev_time}")
        
        # Calculate the derivative
        food_derivative = np.gradient(total_food_i)
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

        # # Second subplot for FPS
        # axs[1].plot(total_fps_i, linewidth=1, color='salmon', linestyle='-', label='FPS')
        # # axs[1].set_xlabel('Frame', fontsize=14)
        # axs[1].set_ylabel('FPS', fontsize=14)
        # # axs[1].set_title('FPS Over Time', fontsize=16)
        # axs[1].axhline(y=average_fps, color='green', linestyle='--', linewidth=1, label='Average FPS')
        # axs[1].legend(frameon=True, loc='upper left')
        # axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)

        # Second subplot for time per iteration
        axs[1].plot(total_time_per_iteration_i, linewidth=1, color='lightcoral', linestyle='-', label='Time per Iteration')
        # axs[1].set_xlabel('Iteration', fontsize=14)
        axs[1].set_ylabel('Time per iteration', fontsize=14)
        # axs[1].set_title('Time per Iteration Over Time', fontsize=16)
        axs[1].axhline(y=average_time_per_iteration, color='green', linestyle='--', linewidth=1, label='Average FPS')
        axs[1].legend(frameon=True, loc='upper left')
        axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)


        # Third subplot for Total Food Collected
        # axs[2].plot(total_food_i, linewidth=1, color='lightgreen', linestyle='-', label='Total Food Collected')
        axs[2].plot(food_derivative[:-1], linewidth=1, color='orange', linestyle='-', label='Food Derivative')
        axs[2].set_xlabel('Iteration', fontsize=14)
        axs[2].set_ylabel('Food Units', fontsize=14)
        # axs[2].set_title('Total Food Collected Over Time', fontsize=16)
        # Add a vertical line at the index
        axs[2].legend(frameon=True, loc='upper left')
        axs[2].grid(True, which='both', linestyle='--', linewidth=0.5)

        # Adjust layout
        plt.tight_layout(pad=3.0)

        # Save the figure
        plt.savefig("combined_plots.png")

        # Show the plot
        plt.show()
        
        
        pg.quit()
        exit(0)