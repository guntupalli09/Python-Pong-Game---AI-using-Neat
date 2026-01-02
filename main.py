from Game import Game
import pygame
import neat
import os
import time
import pickle
import matplotlib.pyplot as plt


class PongGame:
    def __init__(self, window, width, height):
        self.game = Game(window, width, height)
        self.ball = self.game.ball
        self.left_paddle = self.game.left_paddle
        self.right_paddle = self.game.right_paddle

    def test_ai(self, net):
        clock = pygame.time.Clock()
        run = True

        while run:
            clock.tick(60)
            self.game.loop()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break

            output = net.activate((
                self.right_paddle.y,
                abs(self.right_paddle.x - self.ball.x),
                self.ball.y
            ))
            decision = output.index(max(output))

            if decision == 1:
                self.game.move_paddle(left=False, up=True)
            elif decision == 2:
                self.game.move_paddle(left=False, up=False)

            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                self.game.move_paddle(left=True, up=True)
            elif keys[pygame.K_s]:
                self.game.move_paddle(left=True, up=False)

            self.game.draw(draw_score=True)
            pygame.display.update()

    def train_ai(self, genome1, genome2, config, draw=False):
        start_time = time.time()

        net1 = neat.nn.FeedForwardNetwork.create(genome1, config)
        net2 = neat.nn.FeedForwardNetwork.create(genome2, config)

        self.genome1 = genome1
        self.genome2 = genome2

        max_hits = 50

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return True

            game_info = self.game.loop()
            self.move_ai_paddles(net1, net2)

            if draw:
                self.game.draw(draw_score=False, draw_hits=True)

            pygame.display.update()

            duration = time.time() - start_time
            if (
                game_info.left_score == 1
                or game_info.right_score == 1
                or game_info.left_hits >= max_hits
            ):
                self.calculate_fitness(game_info, duration)
                break

        return False

    def move_ai_paddles(self, net1, net2):
        players = [
            (self.genome1, net1, self.left_paddle, True),
            (self.genome2, net2, self.right_paddle, False)
        ]

        for genome, net, paddle, left in players:
            output = net.activate((
                paddle.y,
                abs(paddle.x - self.ball.x),
                self.ball.y
            ))
            decision = output.index(max(output))

            valid = True
            if decision == 0:
                genome.fitness -= 0.01
            elif decision == 1:
                valid = self.game.move_paddle(left=left, up=True)
            else:
                valid = self.game.move_paddle(left=left, up=False)

            if not valid:
                genome.fitness -= 1

    def calculate_fitness(self, game_info, duration):
        self.genome1.fitness += game_info.left_hits + duration
        self.genome2.fitness += game_info.right_hits + duration


def eval_genomes(genomes, config):
    width, height = 700, 500
    win = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Pong")

    for i, (genome_id1, genome1) in enumerate(genomes):
        genome1.fitness = 0

        for genome_id2, genome2 in genomes[min(i + 1, len(genomes) - 1):]:
            genome2.fitness = 0 if genome2.fitness is None else genome2.fitness
            pong = PongGame(win, width, height)

            force_quit = pong.train_ai(genome1, genome2, config, draw=False)
            if force_quit:
                quit()


# ------------------ PLOTTING ------------------

def plot_learning_curves(stats):
    generations = range(len(stats.most_fit_genomes))

    best_fitness = [g.fitness for g in stats.most_fit_genomes]
    mean_fitness = stats.get_fitness_mean()

    plt.figure(figsize=(7, 4))
    plt.plot(generations, best_fitness, label="Best Fitness", linewidth=2)
    plt.plot(generations, mean_fitness, label="Mean Fitness", linewidth=2)

    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("NEAT Training Dynamics: Pong")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig("fitness_vs_generation.png", dpi=150)
    plt.close()


def plot_network_complexity(stats):
    generations = range(len(stats.most_fit_genomes))
    complexity = [len(g.connections) for g in stats.most_fit_genomes]

    plt.figure(figsize=(7, 4))
    plt.plot(generations, complexity, linewidth=2)

    plt.xlabel("Generation")
    plt.ylabel("Number of Connections")
    plt.title("Network Complexity vs Generation")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig("complexity_vs_generation.png", dpi=150)
    plt.close()


# ------------------ RUN NEAT ------------------

def run_neat(config):
    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))

    winner = p.run(eval_genomes, 50)

    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)

    plot_learning_curves(stats)
    plot_network_complexity(stats)


def test_best_network(config):
    with open("best.pickle", "rb") as f:
        winner = pickle.load(f)

    net = neat.nn.FeedForwardNetwork.create(winner, config)

    width, height = 700, 500
    win = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Pong")

    pong = PongGame(win, width, height)
    pong.test_ai(net)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    run_neat(config)
    test_best_network(config)

