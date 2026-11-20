from heapq import heapify, heappop, heappush

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import manhattan_distance, nearest_point


def create_team(first_index, second_index, is_red, num_training=0):
    return [AlmaAgent(first_index), AlmaAgent(second_index, defender=True)]


class AlmaAgent(CaptureAgent):
    def __init__(self, index, defender=False):
        super().__init__(index)
        self.defender = defender
        self.patrol_index = 0
        self.intrusion = None
        self.last_incursion_end = None
        self.beliefs = {}

    def register_initial_state(self, state):
        super().register_initial_state(state)
        self.entry_counts = {}
        self.beliefs = {}
        walls = state.get_walls()
        middle = walls.width // 2
        home, away = (middle - 1, middle) if self.red else (middle, middle - 1)
        # Els passos de frontera permeten dipositar el menjar.
        self.gates = [(home, y) for y in range(1, walls.height - 1)
                      if not walls[home][y] and not walls[away][y]]
        if not self.gates:
            self.gates = [state.get_agent_position(self.index)]
        self.patrol_index = len(self.gates) // 2
        self.neighbors = {}
        for x in range(1, walls.width - 1):
            for y in range(1, walls.height - 1):
                if not walls[x][y]:
                    self.neighbors[x, y] = [(nx, ny) for nx, ny in
                                            ((x-1, y), (x+1, y), (x, y-1), (x, y+1))
                                            if not walls[nx][ny]]
        self.home_cells = [cell for cell in self.neighbors if (cell[0] < middle) == self.red]
        self.exit_by_cell = {
            cell: min(self.gates, key=lambda gate: self.get_maze_distance(cell, gate))
            for cell in self.home_cells
        } if self.defender else {}
        self.food_exit_distance = {pellet: self.distance(pellet, self.gates)
                                   for pellet in self.get_food(state).as_list()}

    def distance(self, position, targets):
        return min((self.get_maze_distance(position, target) for target in targets),
                   default=0)

    def choose_action(self, state):
        own = state.get_agent_state(self.index)
        position = own.get_position()
        opponents = [state.get_agent_state(i) for i in self.get_opponents(state)]
        visible = [enemy for enemy in opponents if enemy.get_position() is not None]
        actions = state.get_legal_actions(self.index)
        lost = set()
        if self.defender:
            previous = self.get_previous_observation()
            if previous is not None:
                lost = set(self.get_food_you_are_defending(previous).as_list())
                lost.difference_update(self.get_food_you_are_defending(state).as_list())
            if previous is not None:
                for index in self.get_opponents(state):
                    enemy = state.get_agent_state(index)
                    old = previous.get_agent_state(index)
                    if enemy.is_pacman and not old.is_pacman and enemy.get_position() in self.gates:
                        gate = enemy.get_position()
                        self.entry_counts[gate] = self.entry_counts.get(gate, 0) + 1
            self.track_invaders(state, position, lost)
        if self.defender and not own.is_pacman:
            if not self.should_counterattack(state, visible):
                return self.defend(state, actions, own, visible, lost)
        return self.attack(state, actions, own, position, visible)

    def should_counterattack(self, state, visible):
        time_left = state.data.timeleft
        previous = self.get_previous_observation()
        if previous is not None and previous.get_agent_state(self.index).is_pacman:
            self.last_incursion_end = time_left
        if (time_left > 520
                or self.get_score(state) >= 0
                or any(enemy.is_pacman for enemy in visible)):
            return False
        # Tornem a defensar durant uns torns després de cada incursió.
        return self.last_incursion_end is None or self.last_incursion_end - time_left >= 48

    def safe_routes(self, goals, ghosts, capsules):
        # Cerca de cost uniforme des dels objectius, penalitzant el perill.
        danger = {}
        for position in self.neighbors:
            distance = self.distance(position, ghosts)
            danger[position] = {0: 70, 1: 28, 2: 9, 3: 3, 4: 1}.get(distance, 0)
        for capsule in capsules:
            danger[capsule] = 0
        costs = dict(goals)
        queue = [(cost, position) for position, cost in goals]
        # Tots els objectius entren a la mateixa cua de prioritat.
        heapify(queue)
        while queue:
            cost, position = heappop(queue)
            if cost != costs[position]:
                continue
            for neighbor in self.neighbors[position]:
                next_cost = cost + 1 + danger[position]
                if next_cost < costs.get(neighbor, float("inf")):
                    costs[neighbor] = next_cost
                    heappush(queue, (next_cost, neighbor))
        return costs

    def attack(self, state, actions, own, position, visible):
        food = self.get_food(state).as_list()
        capsules = self.get_capsules(state)
        ghosts = [enemy.get_position() for enemy in visible
                  if not enemy.is_pacman and enemy.scared_timer <= 2]
        home_distance = self.distance(position, self.gates)
        threatened = ghosts and self.distance(position, ghosts) <= home_distance + 2
        # Amb sortides properes podem allargar la incursió.
        nearby_exits = sum(self.get_maze_distance(position, gate) <= home_distance + 3
                           for gate in self.gates)
        carry_limit = 5
        if (home_distance >= 15 or len(self.gates) == 1) and ghosts:
            carry_limit = 4
        elif home_distance <= 6 and nearby_exits >= 2 and not ghosts:
            carry_limit = 8
        power = min(state.get_agent_state(i).scared_timer
                    for i in self.get_opponents(state))
        # Amb marge per tornar, aprofitem la càpsula per omplir més la càrrega.
        if power >= home_distance + 10:
            carry_limit = max(carry_limit, 9)
        # El menjar només puntua quan l'agent torna a casa.
        retreat = own.num_carrying > 0 and (
            len(food) <= 2 or own.num_carrying >= carry_limit or threatened
            or state.data.timeleft <= 4 * (home_distance + 3)
        )

        routes = None
        if ghosts and self.distance(position, ghosts) <= 7:
            goals = [(gate, 0) for gate in self.gates] if retreat else [
                (pellet, 0.20 * self.food_exit_distance.get(pellet, 0)) for pellet in food
            ]
            if not retreat:
                goals += [(capsule, 0) for capsule in capsules]
            if goals:
                routes = self.safe_routes(goals, ghosts, capsules)

        def value(action):
            successor = state.generate_successor(self.index, action)
            future = successor.get_agent_state(self.index)
            pos = future.get_position()
            score = 24 * (future.num_returned - own.num_returned)
            if own.is_pacman and not future.is_pacman and future.num_returned == own.num_returned:
                score -= 100 + 12 * own.num_carrying
            if routes is not None:
                score -= (2 if retreat else 1) * routes.get(nearest_point(pos), 1000)
                if not retreat:
                    score += 4 * (future.num_carrying - own.num_carrying)
            elif retreat:
                score -= 3 * self.distance(pos, self.gates)
            elif food:
                # Valorem menjar proper amb un camí curt de retorn.
                score -= min(self.get_maze_distance(pos, pellet)
                             + 0.20 * self.food_exit_distance.get(pellet, 0)
                             for pellet in food)
                score += 5 * (future.num_carrying - own.num_carrying)
            remaining_capsules = self.get_capsules(successor)
            if ghosts and capsules:
                score += 18 * (len(capsules) - len(remaining_capsules))
                if not retreat and self.distance(position, ghosts) <= 5:
                    score += 1.4 * (self.distance(position, capsules) - self.distance(pos, capsules))
            if future.is_pacman and ghosts and len(remaining_capsules) == len(capsules):
                clearance = self.distance(pos, ghosts)
                score -= {0: 120, 1: 80, 2: 27, 3: 9, 4: 3}.get(clearance, 0)
            direction = own.configuration.direction
            return score - 0.3 * (action == Directions.REVERSE[direction]) - (action == Directions.STOP)

        return max(actions, key=value)

    def track_invaders(self, state, position, lost):
        readings = state.get_agent_distances()
        teammates = [state.get_agent_position(i) for i in self.get_team(state)]
        hidden_cells = [cell for cell in self.home_cells
                        if all(manhattan_distance(cell, ally) > 5 for ally in teammates)]
        for index in self.get_opponents(state):
            enemy = state.get_agent_state(index)
            if not enemy.is_pacman:
                self.beliefs.pop(index, None)
                continue
            observed = enemy.get_position()
            if observed is not None:
                self.beliefs[index] = {nearest_point(observed): 1.0}
                continue

            # Propaguem l'última creença pels moviments possibles.
            predicted = {}
            for cell, probability in self.beliefs.get(index, {}).items():
                steps = [cell, *self.neighbors.get(cell, ())]
                for step in steps:
                    predicted[step] = predicted.get(step, 0) + probability / len(steps)

            # El sonar filtra posicions i el menjar perdut reforça les properes.
            weights = {}
            for cell in hidden_cells:
                likelihood = state.get_distance_prob(manhattan_distance(position, cell), readings[index])
                if likelihood:
                    clue = 1 + sum(8 / (1 + self.get_maze_distance(cell, pellet)) ** 3 for pellet in lost)
                    weights[cell] = likelihood * (predicted.get(cell, 0) + 0.05 / len(self.home_cells)) * clue
            total = sum(weights.values())
            self.beliefs[index] = {cell: weight / total for cell, weight in weights.items()} if total else {}

    def defend(self, state, actions, own, visible, lost):
        position = own.get_position()
        invaders = [enemy.get_position() for enemy in visible if enemy.is_pacman]
        if lost:
            self.intrusion = min(sorted(lost), key=lambda pos: self.get_maze_distance(position, pos))
        if self.intrusion == position:
            self.intrusion = None
        if position == self.gates[self.patrol_index]:
            self.patrol_index = (self.patrol_index + max(1, len(self.gates) // 2)) % len(self.gates)
        target = self.intrusion or self.gates[self.patrol_index]
        # Alternem la patrulla general amb els passos més usats.
        if not self.intrusion and self.entry_counts and state.data.timeleft % 20 < 12:
            target = max(self.entry_counts, key=self.entry_counts.get)
        if not invaders and not self.intrusion and not any(self.beliefs.values()):
            # Cobrim una entrada només si el rival ja s'hi pot acostar.
            for enemy in visible:
                if enemy.is_pacman:
                    continue
                enemy_pos = enemy.get_position()
                gate = min(self.gates, key=lambda cell: self.get_maze_distance(enemy_pos, cell))
                enemy_distance = self.get_maze_distance(enemy_pos, gate)
                if enemy_distance <= 3 and self.get_maze_distance(position, gate) <= enemy_distance + 1:
                    target = gate
                    break

        if not invaders and own.scared_timer <= 1:
            # Vigilem el pas de fugida probable si encara hi podem arribar.
            gate_weights = {}
            for belief in self.beliefs.values():
                for cell, probability in belief.items():
                    gate = self.exit_by_cell[cell]
                    enemy_distance = self.get_maze_distance(cell, gate)
                    defender_distance = self.get_maze_distance(position, gate)
                    if enemy_distance <= 8 and defender_distance <= enemy_distance + 1:
                        gate_weights[gate] = gate_weights.get(gate, 0) + probability
            if gate_weights:
                gate = max(gate_weights, key=gate_weights.get)
                if gate_weights[gate] >= 0.2:
                    target = gate


        def value(action):
            successor = state.generate_successor(self.index, action)
            future = successor.get_agent_state(self.index)
            pos = nearest_point(future.get_position())
            score = -100 * future.is_pacman
            # Recompensem interceptar abans que l'invasor torni a casa.
            if invaders and own.scared_timer <= 1:
                remaining = sum(successor.get_agent_state(i).is_pacman
                                for i in self.get_opponents(state))
                score += 60 * (sum(state.get_agent_state(i).is_pacman
                                   for i in self.get_opponents(state)) - remaining)
            if invaders:
                distance = self.distance(pos, invaders)
                score -= 3 * (abs(distance - 3) if own.scared_timer > 1 else distance)
            else:
                score -= self.get_maze_distance(pos, target)
            return score - 0.2 * (action == Directions.STOP)

        return max(actions, key=value)
