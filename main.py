import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from math import floor
from copy import deepcopy


source_image = cv2.imread('painting.png')
rows,cols,channels = source_image.shape
num_generations = 10000

#check if circle is in image
def IsInImage(x, y, radii):
   if(x < 0):
      if(y < 0):
         #if radii is longer than the center-closest corner distance return true
         if(x**2 + y**2 <= 2*(radii**2)):
            return True
      elif(0 <= y <= cols):
         #if radii is longer than center's distance to the closest edge return true
         if(x + radii >= 0):
            return True
      else:
         if(x**2 + (y - rows)**2 <= 2*(radii**2)):
            return True
   elif(0 <= x <= cols):
      if(y < 0):
         # if radii is longer than center's distance to the closest edge return true
         if(y + radii >= 0):
            return True
      elif(0 <= y <= rows):
         return True
      else:
         # if radii is longer than center's distance to the closest edge return true
         if (y - radii <= rows):
            return True
   else:
      if(y < 0):
         # if radii is longer than the center-closest corner distance return true
         if((x - cols)**2 + y**2 <= 2*(radii**2)):
            return True
      elif(0 <= y <= rows):
         # if radii is longer than center's distance to the closest edge return true
         if(x - radii >= rows):
            return True
      else:
         # if radii is longer than the center-closest corner distance return true
         if((x - cols)**2 + (y - rows)**2 <= 2*(radii**2)):
            return True
   return False


class Gene:
   def __init__(self, x=0, y=0, radii=0, RED=0, GREEN=0, BLUE=0, ALPHA=0):
      self.x = random.randint(-20, cols + 20)
      self.y = random.randint(-20, rows + 20)
      self.radii = random.randint(1, 40)
      #re-initialize gene until it is in image
      while(not IsInImage(self.x, self.y, self.radii)):
         self.x = random.randint(-10, cols + 10)
         self.y = random.randint(-10, rows + 10)
         self.radii = random.randint(1, 40)
      self.RED = random.randint(0, 255)
      self.GREEN = random.randint(0, 255)
      self.BLUE = random.randint(0, 255)
      self.ALPHA = random.random()

   def mutateGene(self, mutation_type = "guided"):
      first_x = self.x
      first_y = self.y
      first_radii = self.radii
      first_red = self.RED
      first_green = self.GREEN
      first_blue = self.BLUE
      first_alpha = self.ALPHA
      is_radii_good = False
      #guided mutation constraints
      if(mutation_type == "guided"):
         self.x = random.randint(first_x - cols/4, first_x + cols/4)
         self.y = random.randint(first_y - rows/4, first_y + rows/4)

         while(not is_radii_good):
            self.radii = random.randint(first_radii - 10, first_radii + 10)
            if(self.radii > 0):
               is_radii_good = True

         is_radii_good = False
         #check if circle is in image
         while(not IsInImage(self.x, self.y, self.radii)):
            self.x = random.randint(first_x - cols/4, first_x + cols/4)
            self.y = random.randint(first_y - rows/4, first_y + rows/4)
            while(not is_radii_good):
               self.radii = random.randint(first_radii - 10, first_radii + 10)
               if (self.radii > 0):
                  is_radii_good = True

         is_red_good = False
         is_green_good = False
         is_blue_good = False
         #randomly pick new RGB values until they satisfy constraints
         while(not is_red_good):
            self.RED = random.randint(first_red - 64, first_red + 64)
            if(0 <= self.RED <= 255):
               is_red_good = True
         while (not is_green_good):
            self.GREEN = random.randint(first_green - 64, first_green + 64)
            if (0 <= self.GREEN <= 255):
               is_green_good = True
         while (not is_blue_good):
            self.BLUE = random.randint(first_blue - 64, first_blue + 64)
            if (0 <= self.BLUE <= 255):
               is_blue_good = True

         is_alpha_good = False
         #randomly pick new Alpha value until it satisfies constraints
         while(not is_alpha_good):
            self.ALPHA = random.uniform(first_alpha - 0.25, first_alpha + 0.25)
            if (0 <= self.ALPHA <= 1):
               is_alpha_good = True

      elif(mutation_type == "unguided"):
         self.x = random.randint(-20, cols + 20)
         self.y = random.randint(-20, rows + 20)
         self.radii = random.randint(1, 40)
         while(not IsInImage(self.x, self.y, self.radii)):
            self.x = random.randint(-20, cols + 20)
            self.y = random.randint(-20, rows + 20)
            self.radii = random.randint(1, 40)
         self.BLUE = random.randint(0, 255)
         self.GREEN = random.randint(0, 255)
         self.RED = random.randint(0, 255)
         self.ALPHA = random.random()


class Individual:
   def __init__(self, num_genes=50):
      self.fitness = None
      self.num_genes = num_genes
      self.chromosome = [Gene() for i in range(0, num_genes)]

   #mutate individual
   def mutateIndividual(self, mutation_prob=0.2, mutation_type="guided"):
      while (random.random() < mutation_prob):
         mutatedGene = random.randint(0, self.num_genes - 1)
         self.chromosome[mutatedGene].mutateGene(mutation_type)
         self.fitness = None
   #evaluate individual
   def evalIndividual(self, source_image):
      image = np.full((rows, cols, 3), (255, 255, 255), np.uint8)
      for gene in self.chromosome:
         overlay = deepcopy(image)
         cv2.circle(overlay, (gene.x, gene.y), gene.radii, (gene.BLUE, gene.GREEN, gene.RED), thickness=-1)
         cv2.addWeighted(overlay, gene.ALPHA, image, 1 - gene.ALPHA, 0.0, image)

      self.fitness = -np.sum(np.square(np.subtract(source_image.astype(int), image.astype(int))))

   #sort genes according to their radii
   def sortGenes(self):
      self.chromosome.sort(key=lambda x: x.radii, reverse=True)

   def drawImage(self):
      image = np.full((rows, cols, 3), (255, 255, 255), np.uint8)
      self.sortGenes()
      for gene in self.chromosome:
         overlay = deepcopy(image)
         cv2.circle(overlay, (gene.x, gene.y), gene.radii, (gene.BLUE, gene.GREEN, gene.RED), thickness=-1)
         cv2.addWeighted(overlay, gene.ALPHA, image, 1 - gene.ALPHA, 0.0, image)
      return image


class Population:
   def __init__(self, num_inds=20, num_genes=50, tm_size=5, frac_elites=0.2, frac_parents=0.6, mutation_prob=0.2, mutation_type="guided"):
      self.num_inds = num_inds
      self.num_genes = num_genes
      self.tm_size = tm_size
      self.frac_elites = frac_elites
      self.frac_parents = frac_parents
      self.mutation_prob = mutation_prob
      self.mutation_type = mutation_type
      self.individuals = [Individual(num_genes=num_genes) for i in range(0, num_inds)]
      self.elites = []
      self.winners = []
      self.children = []
      #hold non-elites on a list to apply tournament on them
      self.nonElites = []

   #apply evaluation on population
   def evalPopulation(self, src):
      for indiv in self.individuals:
         indiv.evalIndividual(src)

   #population tournament select method
   def pop_tmsel(self):
      # select random individual and initialize it as the best individual
      index = random.randrange(0, len(self.individuals))
      best = self.individuals[index]
      tsize = self.tm_size
      # apply tournament until tm_size is reached
      while(tsize > 0):
         tsize = tsize - 1
         # select a random individual to compare with the current best
         temp_index = random.randrange(0, len(self.individuals))
         next_ind = self.individuals[temp_index]
         # if the new individual is better than the current best, make it best
         if(next_ind.fitness > best.fitness):
            best = next_ind
            index = temp_index

      # remove the winner from the individuals list and return
      self.individuals.pop(index)
      return best

   def selection(self):
      # clear the previous generation
      self.winners.clear()
      self.elites.clear()

      # sort individuals according to their fitness values, so we can easily find the elites
      self.individuals = sorted(self.individuals, key=lambda x: x.fitness, reverse=True)

      # find the elites
      for i in range(floor(self.num_inds * self.frac_elites)):
         temp = deepcopy(self.individuals[i])
         self.elites.append(temp)
      #take elites out from the individuals, so we can apply tournament on remaining
      for i in range(len(self.elites)):
         self.individuals.pop(len(self.individuals) - 1)
      # apply tournament and select winners
      for i in range(floor(self.num_inds * self.frac_parents)):
         self.winners.append(self.pop_tmsel())


      return self.elites[0]



   def crossover(self):
      #clear the previous generation
      self.children.clear()
      self.nonElites.clear()

      while(len(self.winners)):
         #if we have odd number of tournament winners, directly bypass the last remaining winner
         if (len(self.winners) == 1):
            self.children.append(self.winners.pop(0))
         else:
            # choose two individuals randomly from tournament winners to apply corssover
            parent1 = self.winners.pop(random.randrange(0, len(self.winners)))
            parent2 = self.winners.pop(random.randrange(0, len(self.winners)))
            # create children to apply crossover
            children1 = Individual(self.num_genes)
            children2 = Individual(self.num_genes)
            # crossover
            for j in range(0, self.num_genes):
               prob = random.random()
               #if probability is less than 0.5, do not crossover gene
               if(prob < 0.5):
                  children1.chromosome[j] = parent1.chromosome[j]
                  children2.chromosome[j] = parent2.chromosome[j]
               #else apply crossover
               elif(prob > 0.5):
                  children1.chromosome[j] = parent2.chromosome[j]
                  children2.chromosome[j] = parent1.chromosome[j]
            # pass them to next generation
            self.children.append(children1)
            self.children.append(children2)

      self.nonElites = self.individuals + self.children

   # population mutation
   def mutatePopulation(self):
      #only non-elites can be mutated (elites are directly bypassed to next gen)
      for indiv in self.nonElites:
         indiv.mutateIndividual(mutation_prob=self.mutation_prob, mutation_type=self.mutation_type)
      # update individuals
      temp = self.elites + self.nonElites
      self.individuals = deepcopy(temp)

   # sort population according to fitness value
   def sortPopulation(self):
      self.individuals.chromosome = sorted(self.individuals, key=lambda x: x.fitness, reverse=True)

   # sort individuals according to their radii value
   def sortPopRadii(self):
      for indiv in self.individuals:
         indiv.chromosome.sort(key=lambda x: x.radii, reverse=True)

#initialize fitness list for execution
fitness = []

case = 0


current_generation = 1
old_fitness = float('-inf')
fitness.clear()
num_inds = 100
num_genes = 100
tm_size = 10
frac_elites = 0.2
frac_parents = 0.6
mutation_prob = 0.2
mutation_type = "guided"
num_generations = 20000
dir_name = "painting"

pop = Population(num_inds=num_inds, num_genes=num_genes, tm_size=tm_size, frac_elites=frac_elites,frac_parents=frac_parents, mutation_prob=mutation_prob, mutation_type=mutation_type)

while(current_generation <= num_generations):

   pop.sortPopRadii()
   pop.evalPopulation(source_image)
   best = pop.selection()

   print(f'current_generation:{current_generation}, best.fitness:{best.fitness}')

   if (old_fitness > best.fitness):
      print("ERROR")
      break

   fitness = fitness + [best.fitness]
   prev_best = best.fitness
   pop.crossover()
   pop.mutatePopulation()

   if (current_generation % 1000 == 0):
      cv2.imwrite("./" + "__" + dir_name + "__Generation" + str(current_generation) + ".png", best.drawImage())

   # increment the current generation number
   current_generation += 1

plt.figure()
plt.plot(fitness)
plt.savefig("./Results/" + "__" + dir_name + "__1_to_10000__" + str(current_generation) + ".png")

plt.figure()
plt.plot(fitness[999:])
plt.savefig("./Results/" + "__" + dir_name + "__1000_to_10000__" + str(current_generation) + ".png")

