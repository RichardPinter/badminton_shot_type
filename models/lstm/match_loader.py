from itertools import product
from dataclasses import dataclass
import numpy as np
import pandas as pd

import os
import glob
from typing import List, Dict, Optional, Any, Tuple, Union

import logging
from model import ShotClassification

# constants
PREFIXNAME = 'set'
MAINNAME = 'shots'
DETAILNAME = 'wireframe'

@dataclass
class SetShot:
    """Structure to hold the results of a single player shot."""
    shot: int
    shotClass: str
    shotGroup: str
    moves: List[List[Tuple[float,float]]] ## 'x,y' pair, e.g. 14.33,55.332. PlayerSet.pointNames is the name of each pair
   
@dataclass
class PlayerSet:
    player: int
    set: int    
    shots: List[SetShot]
    pointNames: List[str] # names of the elements in shots.moves

@dataclass
class Match:
    name: str
    sets: Dict[int,List[PlayerSet]] # player to list of sets

@dataclass
class ShotType:
    shotClassId: int
    shotClass: str
    shotGroupId: int
    shotGroup: str
    
class MatchLoader:
    """Loads files for the Badminton Player Grading System.

    Provides methods for training the system, predicting shot classes,
    and saving/loading the trained state.

    Attributes:
        config_path (str): Path to the configuration file.
        logger (str): the logger
    """
    def __init__(self, config_path: str, logger: logging.Logger):
        self.config_path = config_path
        self.logger = logger
        self.includePointNames:List[str] = None
        self.ignoreShotClasses:hash[str] = None
        self.ignoreShotGroups:hash[str] = None
        self.ignoreShotPlayer:hash[str] = None
        self.maxMovesPerShot:int = 40

    # the names of the points. The first is considered the template, the others must all match
    _pointNames: list[str] = None

    def _getEffectiveColNames(self, headerRows:Any) -> Tuple[List[str],List[str]]:
        
        xyCols = list(headerRows.columns[headerRows.columns.str.contains(pat = '\_X|\_Y', regex=True)].values)
        otherCols = [item for item in list(headerRows.columns) if item not in xyCols]

        xyNames = list(np.unique([x[:-2] for x in xyCols]))
        xyNames.sort()

        useXyNames = list(xyNames)
        if self.includePointNames is not None:
            useXyNames = list(set(self.includePointNames) & set(xyNames))

        keepCols = list[str](otherCols)

        for useXyName in useXyNames:
            keepCols.append(useXyName + '_X')
            keepCols.append(useXyName + '_Y')

        if(self._pointNames is None):
            self._pointNames = useXyNames
        elif(useXyNames != self._pointNames):
            raise Exception(f"Point names mismatch, expected [{self._pointNames}] got [{useXyNames}]")
        
        return (keepCols,useXyNames)
    
    def loadPlayerSets(self, setid:int, mainpath:str, detailpath:str) -> List[PlayerSet]:
                
        dataMain = pd.read_csv(mainpath, usecols=['player', 'shot', 'shot_class', 'shot_class_grouped'])
                
        headerRows = pd.read_csv(detailpath, nrows=1)
        keepCols, useXyNames = self._getEffectiveColNames(headerRows)

        dataDetail = pd.read_csv(detailpath, usecols = keepCols )
        
        dctPlayer = dict[int,PlayerSet]()
        
        for group in dataDetail.groupby(by=['Player', 'shot']).groups:
                
                playerStr = group[0]
                if not self.ignoreShotPlayer is None and playerStr in self.ignoreShotPlayer:
                    continue

                player = 0
                if playerStr == 'A': player = 1
                elif playerStr == 'B': player = 2
                
                shot = group[1]

                playerSet = dctPlayer.get(player)
                if playerSet is None:
                    playerSet = PlayerSet(player,setid,[],useXyNames)
                    dctPlayer[playerSet.player] = playerSet
                
                itemMain = dataMain.loc[(dataMain['player'] == playerStr) & (dataMain['shot'] == shot)].head(1)
                idx = itemMain.index[0]

                shotClass = itemMain['shot_class'][idx]

                if not self.ignoreShotClasses is None and shotClass in self.ignoreShotClasses:
                    continue

                shotGroup = itemMain['shot_class_grouped'][idx]
                if not self.ignoreShotGroups is None and shotGroup in self.ignoreShotGroups:
                    continue

                setShot = SetShot(shot, shotClass, shotGroup, list[List[Tuple[float,float]]]())
                playerSet.shots.append(setShot)

                itemDetails = dataDetail.loc[(dataDetail['Player'] == playerStr) & (dataDetail['shot'] == shot)]
                
                count = 0
                for index, row in itemDetails.iterrows():
                    setShot.moves.append(list(map(lambda n: (float(row[n + '_X']),float(row[n + '_Y'])), useXyNames)))
                    count += 1
                    if count > self.maxMovesPerShot:
                        break

                if len(setShot.moves) == 0:
                    raise f"Error - no moves present for player {playerStr} shot {shot}"

        return list(dctPlayer.values())

                    
    def loadMatches(self, filepath: str, limit:int = -1) -> List[Match]:
        
        matches = dict[str,Match]()

        mainFilter = os.path.join(os.getcwd(), os.path.normpath(filepath), '*', f'{PREFIXNAME}*_{MAINNAME}.csv')
        
        for shotPath in glob.iglob(mainFilter):
            match: Match
           
            pathSegments = os.path.split(shotPath)
            dir = pathSegments[0]
            file = pathSegments[1]
            name = os.path.split(dir)[1] # dir the file resides in
            setid = int(file.removeprefix(PREFIXNAME).removesuffix(f"_{MAINNAME}.csv"))

            match = matches.get(name)
            if match is None:
                match = Match(name, dict[int,List[PlayerSet]]())
                matches[name] = match
            
            wireframePath = os.path.abspath(f'{shotPath}/../{os.path.basename(shotPath).replace(MAINNAME,DETAILNAME)}')
            sets: List[PlayerSet] = self.loadPlayerSets(setid, shotPath, wireframePath)
            for set in sets:
                lst = match.sets.get(set.player)
                if lst is None:
                    lst = list[PlayerSet]()
                    match.sets[set.player] = lst
                lst.append(set)
            
            if limit > -1 and len(matches) >= limit:
                break

        result = list(matches.values())

        if(len(result) == 0):
            self.logger.error(f"No matches loaded from {filepath}")

        return result
    
class MatchUtils:
    @staticmethod
    def AllShots(matches:List[Match]) -> List[SetShot]:
        result = list[SetShot]()
        for match in matches:
            for lstPlayerSets in list(match.sets.values()):
                for playerSet in lstPlayerSets:
                    for shot in playerSet.shots:
                        result.append(shot)
        return result
    
    @staticmethod
    def Summary(matches:List[Match]) -> Dict[str,str]:
        summary = dict[str,str]()
        summary["MatchCount"] = len(matches)
        pointNames = set()
        setNumbers = set()
        shotGroupCounts = dict[str,int]()
        totalPlayers = 0
        for match in matches:
            matchPlayers = len(match.sets.keys())
            if matchPlayers > totalPlayers:
                totalPlayers = matchPlayers

            for listPlayerSet in list(match.sets.values()):
                for playerSet in listPlayerSet:
                    for pointName in playerSet.pointNames:
                        pointNames.add(pointName)
                    setNumbers.add(playerSet.set)
                    for shot in playerSet.shots:
                        if shotGroupCounts.get(shot.shotGroup) is None:
                            shotGroupCounts[shot.shotGroup] = 0
                        shotGroupCounts[shot.shotGroup] = shotGroupCounts.get(shot.shotGroup) + 1
        
        summary["PlayerCount"] = str(totalPlayers)
        summary["PointNames"] = ",".join(list(pointNames))
        summary["SetNumbers"] = ",".join(map(str, setNumbers))
        totalShotCount = 0
        for shotGroupName, shotGroupCount in shotGroupCounts.items():
            summary["ShotGroup:" + shotGroupName] = str(shotGroupCount)
            totalShotCount += shotGroupCount

        summary["ShotCount"] = str(totalShotCount)
        return summary   

    @staticmethod
    def PrintSummary(matches:List[Match]):
        summary = MatchUtils.Summary(matches)

        print('Name            Value')
        for key, val in summary.items():
            print('{} {}'.format(key, val))


class ShotUtils:
    @staticmethod
    def AllMoves(set_shot:SetShot, floatToIntFactorX:float, floatToIntFactorY:float) -> List[List[List[int]]]:
        result = list[List[List[int]]]()
        for move in set_shot.moves:
             resultMove = list[List[int]]()
             for point in move:
                 resultPoint = list[int]()
                 resultPoint.append(int(point[0] * floatToIntFactorX))
                 resultPoint.append(int(point[1] * floatToIntFactorY))
                 resultMove.append(resultPoint)
             result.append(resultMove)
        return result
    
    @staticmethod
    def LoadShotTypes(filepath:str) -> List[ShotType]:
        groupIndex: int = 0
        groupNamesToId:Dict[str,int] = dict[str,int]()
        
        shotTypeList = list[ShotType]()

        cols = ['English', 'EnglishFrontiers']
        dataMain = pd.read_csv(filepath, usecols=cols)
        
        for index, row in dataMain.iterrows():
            shotClassName = row[cols[0]]
            shotClassIndex = index
            shotGroupName = row[cols[1]]
            if groupNamesToId.get(shotGroupName) is None:
                groupNamesToId[shotGroupName] = groupIndex
                groupIndex += 1

            shotTypeList.append(ShotType(shotClassIndex, shotClassName, groupNamesToId.get(shotGroupName), shotGroupName))

        return shotTypeList
    
    @staticmethod
    def ToDictGroupNameToId(shotTypes:List[ShotType]) -> dict[str,int]:
        shotGroupToId = dict[str,int]()
        for shotType in shotTypes:
            if shotGroupToId.get(shotType.shotGroup) is None:
                shotGroupToId[shotType.shotGroup] = shotType.shotGroupId
        return shotGroupToId
    
    @staticmethod
    def ToDictGroupIdToName(shotTypes:List[ShotType]) -> dict[int,str]:
        shotGroupIdToName = dict[int,str]()
        for shotType in shotTypes:
            if shotGroupIdToName.get(shotType.shotGroupId) is None:
                shotGroupIdToName[shotType.shotGroupId] = shotType.shotGroup
        return shotGroupIdToName

