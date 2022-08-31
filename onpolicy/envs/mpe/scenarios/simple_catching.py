from tabnanny import check
import numpy as np
from onpolicy.envs.mpe.core import World, Agent, Landmark
from onpolicy.envs.mpe.scenario import BaseScenario
from PIL import Image
import os
import cv2
import skimage.morphology

class ExpWorld(World):
    def __init__(self, args):
        super().__init__()
        self.maps_path = args.maps_path
        self.trav_map_default_resolution = args.trav_map_default_resolution
        self.trav_map_resolution = args.trav_map_resolution
        self.trav_map = self.load_trav_map()
    
    def load_trav_map(self):
        #  Loads the traversability maps for all floors
        # Todo：modify the image file path 
        trav_map = np.array(Image.open(self.maps_path))
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
        # inflate the obstacle
        contact_margin = entity.size/2
        selem = skimage.morphology.disk(int(contact_margin/self.trav_map_resolution))
        obstacle_grid = skimage.morphology.binary_dilation(self.trav_map, selem)
        entity_index = self.world_to_grid(entity.state.p_pos)

        # check if colliding with obstacle
        if_collide = False
        x1 = max(0, entity_index[0]-2)
        x2 = min(obstacle_grid.shape[0]-1, entity_index[0]+2)
        y1 = max(0, entity_index[1]-2)
        y2 = min(obstacle_grid.shape[1]-1, entity_index[1]+2)
        if np.sum(obstacle_grid[x1:x2,y1:y2])>10:
            if_collide = True
        return if_collide

    def update_agent_state(self, agent):
        super().update_agent_state(agent)
        agent.grid_index = self.world_to_grid(agent.state.p_pos)
        agent.if_collide = self.check_obstacle_collision(agent)
        
    


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
            agent.size = 0.2 if agent.adversary else 0.15
            agent.accel = 3.0 if agent.adversary else 4.0
            #agent.accel = 20.0 if agent.adversary else 25.0
            agent.max_speed = 1.0 if agent.adversary else 1.3
            agent.grid_index = None

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
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.grid_index = self.world_to_grid(agent.state.p_pos, world)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = 0.8 * np.random.uniform(-1, +1, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)
    

    def world_to_grid(self, p_pos, world):
        # Transform the coordinate in the world to the index in the travesable map
        trav_map_size = world.trav_map.shape[0]
        x = int(trav_map_size//2 - p_pos[1]/world.trav_map_resolution)
        y = int(trav_map_size//2 + p_pos[0]/world.trav_map_resolution)

        return np.array([x,y], dtype=int) 

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
        shape = False #different from openai
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (increased reward for increased distance from adversary)
            for adv in adversaries:
                rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 10

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        shape = False #different from openai
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (decreased reward for increased distance from agents)
            for adv in adversaries:
                rew -= 0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in agents])
        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        rew += 10
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
    env = CatchingEnv(world, scenario.reset_world,
                        scenario.reward, scenario.observation, scenario.info)
    
    env.reset()
    for i in range(20):
        action = [[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0]]
        env.step(action)
        env.render(save_path="/workspace/tmp")

    print("done")

if __name__=="__main__":
   main()



