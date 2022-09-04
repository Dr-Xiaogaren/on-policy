from re import S
from tabnanny import check
from tkinter import W
import numpy as np
from onpolicy.envs.mpe.core import World, Agent, Landmark
from onpolicy.envs.mpe.scenario import BaseScenario
from PIL import Image
import os
import cv2
import skimage.morphology
import random
import math
import imageio
import copy
class ExpWorld(World):
    def __init__(self, args):
        super().__init__()
        self.maps_path = args.maps_path
        self.trav_map_default_resolution = args.trav_map_default_resolution
        self.trav_map_resolution = args.trav_map_resolution
        self.trav_map = self.load_trav_map(self.maps_path)
        self.min_initial_distance = args.min_initial_distance
        self.max_initial_inner_distance = args.max_initial_inner_distance
        self.max_initial_inter_distance = args.max_initial_inter_distance
    
    def load_trav_map(self, maps_path):
        #  Loads the traversability maps for all floors
        # Todo：modify the image file path 
        trav_map = np.array(Image.open(maps_path))
        height, width = trav_map.shape
        assert height == width, "trav map is not a square"
        trav_map_original_size = height
        trav_map_size = int(
            trav_map_original_size * self.trav_map_default_resolution / self.trav_map_resolution
        )
        # We resize the traversability map to the new size computed before
        trav_map = cv2.resize(trav_map, (trav_map_size, trav_map_size))
        # We make the pixels of the image to be either 0 or 1
        trav_map[trav_map < 255] = 0
        trav_map[trav_map > 0] = 1
        trav_map = 1-trav_map
        return trav_map

    
    def world_to_grid(self, p_pos):
        # Transform the coordinate in the world to the index in the travesable map
        trav_map_size = self.trav_map.shape[0]
        x = int(trav_map_size//2 - p_pos[1]/self.trav_map_resolution)
        y = int(trav_map_size//2 + p_pos[0]/self.trav_map_resolution)

        return np.array([x,y], dtype=int) 

    def check_obstacle_collision(self, entity):
        # check if collide with obstacle
        # inflate the obstacle
        contact_margin = entity.size
        selem = skimage.morphology.disk(int(contact_margin/self.trav_map_resolution))
        obstacle_grid = skimage.morphology.binary_dilation(self.trav_map, selem)
        entity_index = self.world_to_grid(entity.state.p_pos)

        # check if colliding with obstacle, collide threthold is 1 grid
        if_collide = False
        x1 = max(0, entity_index[0]-1)
        x2 = min(obstacle_grid.shape[0]-1, entity_index[0]+1)
        y1 = max(0, entity_index[1]-1)
        y2 = min(obstacle_grid.shape[1]-1, entity_index[1]+1)
        if np.sum(obstacle_grid[x1:x2,y1:y2])>2:
            if_collide = True
        return if_collide
    
    def check_if_dead(self, entity):
        # check if agent is no way out
        pos = np.copy(entity.state.p_pos)
        shift_degree = [0, math.pi/6, math.pi/3, math.pi/2, 2*math.pi/3, 5*math.pi/6, 
                       math.pi, 7*math.pi/6, 3*math.pi/2, 11*math.pi/6]
        shift_margin = 0.15
        collide_num = 0
        for shift in shift_degree:
            entity.state.p_pos = pos +  np.array([math.sin(shift),math.cos(shift)])*shift_margin
            if self.check_obstacle_collision(entity):
                collide_num += 1
                continue
            else:
                for a in self.agents:
                    if self.check_agent_collision(a, entity) and a.name!=entity.name:
                        collide_num += 1
                        continue
        entity.state.p_pos = pos

        return True if collide_num == len(shift_degree) else False

    

    def check_agent_collision(self, agent1, agent2):
        # check if collide witth agent
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False
    
    def update_agent_state(self, agent):
        super().update_agent_state(agent)
        agent.grid_index = self.world_to_grid(agent.state.p_pos)
        agent.if_collide = self.check_obstacle_collision(agent)

        # update state of the world
    def step(self):
        # zoe 20200420
        self.world_step += 1
        # set actions for scripted agents
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        rotation = [None] * len(self.entities)
        # apply agent physical controls
        rotation = self.apply_rotation(rotation)
        # Rotation is performed first   
        self.integrate_rotation(rotation)

        p_force = self.apply_action_force(p_force)
        # apply environment forces
        p_force = self.apply_environment_force(p_force)
        # integrate physical state
        self.integrate_state(p_force)
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)
        # calculate and store distances between all entities
        if self.cache_dists:
            self.calculate_distances()
    
    def apply_rotation(self, rotation):
        # return the rotation angle
        for i, agent in enumerate(self.agents):
            if agent.movable:
                # force = mass * a * action + n
                if agent.action.u[1] > 0:
                    rotation[i] = agent.rotation_stepsize
                elif agent.action.u[1] < 0:
                    rotation[i] = -agent.rotation_stepsize
                elif agent.action.u[1] == 0:
                    rotation[i] = 0
    
        return rotation
    
    def integrate_rotation(self, rotation):
        # perform rotation
        for i, agent in enumerate(self.agents):
            if not agent.movable:
                continue
            agent.orientation = agent.orientation + rotation[i]
            agent.orientation = agent.orientation % (2*np.math.pi)


    def apply_action_force(self, p_force):
        # set applied forces
        for i, agent in enumerate(self.agents):
            if agent.movable:
                noise = np.random.randn(
                    *agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                # force = mass * a * action + n
                orientation = np.array([math.cos(agent.orientation), math.sin(agent.orientation) ])
                p_force[i] = (
                    agent.mass * agent.accel if agent.accel is not None else agent.mass) * agent.action.u[0] * orientation + noise
        return p_force

    def integrate_state(self, p_force):
        for i, entity in enumerate(self.entities):
            if not entity.movable:
                continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if (p_force[i] is not None):
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(
                    np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                      np.square(entity.state.p_vel[1])) * entity.max_speed
            # if the action won't make agent get rid of collision, agent will not move anymore
            old_entity_pos = np.copy(entity.state.p_pos)
            entity.state.p_pos += entity.state.p_vel * self.dt
            if entity.if_collide and self.check_obstacle_collision(entity):
                entity.state.p_pos = old_entity_pos
                entity.state.vel = 0
    


class Scenario(BaseScenario):
    def make_world(self, args):
        world = ExpWorld(args)
        # set any world properties first
        world.dim_c = 2
        num_good_agents = args.num_good_agents#1
        num_adversaries = args.num_adversaries#3
        num_agents = num_adversaries + num_good_agents
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.if_collide = False
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.2 if agent.adversary else 0.2
            agent.accel = 3.0 if agent.adversary else 4.0
            #agent.accel = 20.0 if agent.adversary else 25.0
            agent.max_speed = 1.0 if agent.adversary else 1.3
            agent.grid_index = None
            agent.orientation = 0  # pi , The angle with the x-axis, counterclockwise is positive
            agent.rotation_stepsize = math.pi/12
            agent.last_pos = None # pos in last time step

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        world.assign_agent_colors()
        # random properties for landmarks
        world.assign_landmark_colors()
        world.world_step = 0
        # random properties for landmarks
        # set random initial states
        initial_pos = []
        travel_map = world.trav_map
        travel_map_revolution = world.trav_map_resolution
        # the passable index
        selem = skimage.morphology.disk(int(world.agents[0].size/travel_map_revolution))
        obstacle_grid = skimage.morphology.binary_dilation(travel_map, selem)
        index_travelable = np.where(obstacle_grid == 0)
        for agent in world.agents:
            if agent.adversary:
                # random choose
                choose = random.randrange(0, index_travelable[0].shape[0])
                # choose index
                choose_index = np.array([index_travelable[0][choose],index_travelable[1][choose]])
                choose_pos = self.grid_to_world(choose_index, world)

                agent.state.p_pos = choose_pos
                min_distance = min([np.linalg.norm(choose_pos-loc) for loc in initial_pos]) \
                        if len(initial_pos) != 0 else world.min_initial_distance
                max_distance = max([np.linalg.norm(choose_pos-loc) for loc in initial_pos[0:]]) \
                        if len(initial_pos) != 0 else world.max_initial_inner_distance

                while world.check_obstacle_collision(agent) or world.min_initial_distance > min_distance or world.max_initial_inner_distance < max_distance:
                    choose = random.randrange(0, index_travelable[0].shape[0])
                    choose_index = np.array([index_travelable[0][choose],index_travelable[1][choose]])
                    choose_pos = self.grid_to_world(choose_index, world)
                    min_distance = min([np.linalg.norm(choose_pos-loc) for loc in initial_pos]) \
                        if len(initial_pos) != 0 else world.min_initial_distance
                    max_distance = max([np.linalg.norm(choose_pos-loc) for loc in initial_pos[0:]]) \
                        if len(initial_pos) != 0 else world.max_initial_inner_distance
                    agent.state.p_pos = choose_pos

                initial_pos.append(choose_pos)

            else:
                # random choose
                choose = random.randrange(0, index_travelable[0].shape[0])
                # choose index
                choose_index = np.array([index_travelable[0][choose],index_travelable[1][choose]])
                choose_pos = self.grid_to_world(choose_index, world)
                agent.state.p_pos = choose_pos
                min_distance = min([np.linalg.norm(choose_pos-loc) for loc in initial_pos]) \
                        if len(initial_pos) != 0 else world.min_initial_distance
                max_distance = max([np.linalg.norm(choose_pos-loc) for loc in initial_pos[0:]]) \
                        if len(initial_pos) != 0 else world.max_initial_inter_distance
                while world.check_obstacle_collision(agent) or world.min_initial_distance > min_distance or world.max_initial_inter_distance < max_distance:
                    choose = random.randrange(0, index_travelable[0].shape[0])
                    choose_index = np.array([index_travelable[0][choose],index_travelable[1][choose]])
                    choose_pos = self.grid_to_world(choose_index, world)
                    min_distance = min([np.linalg.norm(choose_pos-loc) for loc in initial_pos]) \
                        if len(initial_pos) != 0 else world.min_initial_distance
                    max_distance = max([np.linalg.norm(choose_pos-loc) for loc in initial_pos[0:]]) \
                        if len(initial_pos) != 0 else world.max_initial_inter_distance
                    agent.state.p_pos = choose_pos

            # other massage
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.grid_index = self.world_to_grid(agent.state.p_pos, world)
            # last pose
            agent.last_pos = agent.state.p_pos
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = 0.8 * np.random.uniform(-1, +1, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)
    

    def world_to_grid(self, p_pos, world):
        # Transform the coordinate in the world to the index in the travesable map
        trav_map_size = world.trav_map.shape[0]
        x = int(trav_map_size//2 - p_pos[1]/world.trav_map_resolution)
        y = int(trav_map_size//2 + p_pos[0]/world.trav_map_resolution)

        return np.array([x, y], dtype=int) 

    def grid_to_world(self, grid_index, world):
        # Transform the index in the travesable map to the coordinate in the world
        trav_map_size = world.trav_map.shape[0]
        pos_0 = (grid_index[1] - trav_map_size//2)*world.trav_map_resolution
        pos_1 = (trav_map_size//2 - grid_index[0])*world.trav_map_resolution

        return np.array([pos_0, pos_1], dtype=float) 


    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]


    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        adversaries = self.adversaries(world)
        # reward according to distance
        for adv in adversaries:
            diff_distance =np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos))) \
                            - np.sqrt(np.sum(np.square(agent.last_pos - adv.last_pos)))
            rew +=  diff_distance
        # if catch
        for a in adversaries:
            if self.is_collision(a, agent):
                rew -= 10

        # if collide
        if agent.if_collide:
            rew += -5

        return rew

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        # reward according to distance
        for ag in agents:
            diff_distance =np.sqrt(np.sum(np.square(agent.state.p_pos - ag.state.p_pos))) \
                            - np.sqrt(np.sum(np.square(agent.last_pos - ag.last_pos)))
            rew -=  diff_distance
        # if catch
        for ag in agents:
            for adv in adversaries:
                if self.is_collision(ag, adv):
                    rew += 10
        # if collide
        if agent.if_collide:
            rew += -5

        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            if not other.adversary:
                other_vel.append(other.state.p_vel)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)

    # change variables after reward function
    def post_step(self, world):
        for agent in world.agents:
            agent.last_pos = agent.state.p_pos
    

    def if_done(self, agent, world):
        agents = self.good_agents(world)
        done = 0
        for a in agents:
            if world.check_if_dead(a):
                done = 1

        return done



def main():
    from onpolicy.envs.mpe.environment import MultiAgentEnv, CatchingEnv
    from onpolicy.envs.mpe.scenarios import load
    from onpolicy.config import get_config
    parser = get_config()
    args = parser.parse_known_args()[0]
    args.env_name = "MPE"
    args.scenario_name = "simple_catching"
    scenario = load(args.scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world(args)
    # create multiagent environment
    env = CatchingEnv(world, reset_callback=scenario.reset_world, reward_callback=scenario.reward, 
                        observation_callback= scenario.observation, info_callback=  scenario.info, 
                        done_callback=scenario.if_done, post_step_callback=scenario.post_step)
    
    # env.reset()
    frames = []
    for i in range(50):
        one_action = [0,0,1,0,0]
        action = []
        for j in range(4):
            # random.shuffle(one_action)
            action.append(copy.copy(one_action))
        obs_n, reward_n, done_n, info_n = env.step(action)
        img = env.render()
        frames.append(img)
        cv2.imwrite("/workspace/tmp/image/{}.png".format(str(i)), img)
        # imageio.mimsave("/workspace/tmp/test.gif", frames, 'GIF', duration=0.1)

    print("done")

if __name__=="__main__":
   main()



