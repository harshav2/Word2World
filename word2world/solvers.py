'''
Credit to: Ahmed Khalifa, https://github.com/amidos2006/AutoSokoban/blob/master/Sokoban/Sokoban.py
'''

import numpy as np
from queue import PriorityQueue
import time
from tqdm import tqdm


class State:
    def __init__(self):
        self.solid=[]
        self.deadlocks = []
        self.targets=[]
        self.crates=[]
        self.player=None
        

    def randomInitialize(self, width, height):
        self.width-width
        self.height=height

        return

    def stringInitialize(self, lines):
        self.solid=[]
        self.targets=[]
        self.crates=[]
        self.player=None

        # clean the input
        for i in range(len(lines)):
            lines[i]=lines[i].replace("\n","")

        for i in range(len(lines)):
            if len(lines[i].strip()) != 0:
                break
            else:
                del lines[i]
                i-=1
        for i in range(len(lines)-1,0,-1):
            if len(lines[i].strip()) != 0:
                break
            else:
                del lines[i]
                i+=1

        #get size of the map
        self.width=0
        self.height=len(lines)
        for l in lines:
            if len(l) > self.width:
                self.width = len(l)

        #set the level
        for y in range(self.height):
            l = lines[y]
            self.solid.append([])
            for x in range(self.width):
                if x > len(l)-1:
                    self.solid[y].append(False)
                    continue
                c=l[x]
                if c == "#":
                    self.solid[y].append(True)
                else:
                    self.solid[y].append(False)
                    if c == "@" or c=="+":
                        self.player={"x":x, "y":y}
                    if c=="$" or c=="*":
                        self.crates.append({"x":x, "y":y})
                    if c=="." or c=="+" or c=="*":
                        self.targets.append({"x":x, "y":y})
        self.intializeDeadlocks()

        return self

    def clone(self):
        clone=State()
        clone.width = self.width
        clone.height = self.height
        # since the solid is not changing then copy by value
        clone.solid = self.solid
        clone.deadlocks = self.deadlocks
        clone.player={"x":self.player["x"], "y":self.player["y"]}

        for t in self.targets:
            clone.targets.append({"x":t["x"], "y":t["y"]})

        for c in self.crates:
            clone.crates.append({"x":c["x"], "y":c["y"]})

        return clone
    
    def checkOutside(self, x, y):
        pass

    def checkMovableLocation(self, x, y):
        pass

    def checkWin(self):
        pass

    def getHeuristic(self):
        targets=[]
        for t in self.targets:
            targets.append(t)
        distance=0
        for c in self.crates:
            bestDist = self.width + self.height
            bestMatch = 0
            for i,t in enumerate(targets):
                if bestDist > abs(c["x"] - t["x"]) + abs(c["y"] - t["y"]):
                    bestMatch = i
                    bestDist = abs(c["x"] - t["x"]) + abs(c["y"] - t["y"])
            distance += abs(targets[bestMatch]["x"] - c["x"]) + abs(targets[bestMatch]["y"] - c["y"])
            del targets[bestMatch]
        return distance

    def update(self, dirX, dirY):
        pass

    def getKey(self):
        key=str(self.player["x"]) + "," + str(self.player["y"]) + "," + str(len(self.crates)) + "," + str(len(self.targets))
        for c in self.crates:
            key += "," + str(c["x"]) + "," + str(c["y"])
        for t in self.targets:
            key += "," + str(t["x"]) + "," + str(t["y"])
        return key

    def __str__(self):
        result = ""
        for y in range(self.height):
            for x in range(self.width):
                if self.solid[y][x]:
                    result += "#"
                else:
                    crate=self.checkCrateLocation(x,y) is not None
                    target=self.checkTargetLocation(x,y) is not None
                    player=self.player["x"]==x and self.player["y"]==y
                    if crate:
                        if target:
                            result += "*"
                        else:
                            result += "$"
                    elif player:
                        if target:
                            result += "+"
                        else:
                            result += "@"
                    else:
                        if target:
                            result += "."
                        else:
                            result += " "
            result += "\n"
        return result[:-1]
    
