# src/shot_classifier.py
"""
Shot Classifier Module for the Badminton Player Grading System.

Provides the ShotClassifier class, which trains on annotated movements to recognise shots

Author: Drew
Role: Machine Learning Software Developer
"""

import tensorflow as tf 
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense,MaxPooling1D,Softmax, LSTM, Dropout, Masking
from tensorflow.keras import utils

from dataclasses import dataclass
import numpy as np
import pandas as pd
import joblib
import os
from typing import List, Dict, Optional, Any, Tuple, Union
import glob

import sys
import logging
import shutil
from src.model import ShotClassification
from src.match_loader import Match, PlayerSet, SetShot, ShotType, MatchUtils, ShotUtils

@dataclass
class ModelSetShot:
    classId: int
    data: List[List[List[int]]] # shotMovements [points [xy pairs]]

class ShotClassifier:
    """High-level ShotClassifier for the Badminton Player Grading System.

    Provides methods for training the system, predicting shot classes,
    and saving/loading the trained state.

    Attributes:
        config_path (str): Path to the configuration file.
    """

    def __init__(self, logger: logging.Logger, shot_types: List[ShotType], model_persist_path: str, config_path: str = 'src/config.json'):
        """Initializes the ShotClassifier.

        Initializes all components.
        If `model_persist_path` is provided and valid, it attempts to load the saved API state using `load_state`.

        Args:
            model_persist_path: Optional path to a saved API state file (.joblib).
            config_path: Path to the configuration file (used for ConativeFramework
                and potentially saved/loaded with state).

        Raises:
            ConfigurationError: If ConativeFramework fails to initialize due to config issues.
            GradingSystemError: For other unexpected initialization errors.
            (Exceptions from `load_state` if loading fails, e.g., FileNotFoundError,
             ModelError, ConfigurationError).
        """
        self.config_path = config_path
        self.logger = logger
        self.model_path = model_persist_path
        self.shot_types = shot_types
        self.floatToIntFactor = (15,15)
        
    "Train the model using shots from multiple players/matches"
    def train(self, name:str, matches: List[Match]):
        lstModelEntries = self._matchesToData(matches)
        return self._trainModel(name,lstModelEntries)

    "Predict a shot group using the named model"
    def predict(self, modelPath:str, set_shots: List[SetShot]) -> List[str]:
        model = load_model(modelPath, compile=False)
        result = list[str]()

        dictGroupIdToName = ShotUtils.ToDictGroupIdToName(self.shot_types)

        modelSetShots = self._setshotsToData(set_shots)

        for modelSetShot in modelSetShots:
            predictedGroupId = self._predictModel(model, modelSetShot.data)
            result.append(dictGroupIdToName[predictedGroupId])

        return result
                   
    def _trainModel(self, name:str, modelEntries: List[ModelSetShot], epochs:int = 30) -> str:
        
        classes = list[int]()
        data = list[List[List[List[int]]]]()
        
        for ModelSetShot in modelEntries:
            classes.append(ModelSetShot.classId)
            data.append(ModelSetShot.data)

        allClasses = list(ShotUtils.ToDictGroupNameToId(self.shot_types).values())
        density = max(allClasses)-min(allClasses)+1

        shapedData = np.asarray(data)
        shapedData = np.reshape(shapedData, (shapedData.shape[0], shapedData.shape[1],shapedData.shape[2]*shapedData.shape[3]))
        shapedData = shapedData/np.max(shapedData)
        print(shapedData.shape)

        input_shape = (shapedData.shape[1], shapedData.shape[2])

        model = Sequential()
        model.add(Masking(mask_value = 0, input_shape = input_shape))
        model.add(LSTM(256, input_shape = input_shape, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(256))
        model.add(Dropout(0.2))
        model.add(Dense(density, activation='softmax'))
        model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.summary()
        filepath = os.path.join(self.model_path, f"{name}-" + "{epoch:02d}-{accuracy:.4f}-{loss:.4f}.keras")
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose = 1, save_best_only=True, mode = 'min')
        callbacks_list = [checkpoint]

        categories = utils.to_categorical(classes)
        fitResult = model.fit(shapedData, categories, epochs=400, batch_size=100, callbacks=callbacks_list)

        finalModelFile = list(glob.iglob(checkpoint.filepath.format(epoch=8888, accuracy=9999, loss=checkpoint.best).replace('8888', '*').replace('9999.0000', '*')))[0]

        resultFile = os.path.join(self.model_path, f"{name}.keras")
        shutil.copy(finalModelFile, resultFile)
        return resultFile
    
    def _setshotsToData(self, set_shots: List[SetShot]) -> List[ModelSetShot]:
        # some have different move counts - e.g. a smash is fast and might have 18 moves, while a lob might have 30,
        # so we need to pad the shorter ones out so each shot has the same moves count and mask them out in the model
        maxMoves = 0
        for shot in set_shots:
            moves = len(shot.moves)
            if moves > maxMoves:
                maxMoves = moves

        moveLen = len(set_shots[0].moves[0]) # number of points in a move, e.g nose, wrist, etc

        empty = [[0,0]] * moveLen
        
        result = list[ModelSetShot]()

        shotGroupToId = ShotUtils.ToDictGroupNameToId(self.shot_types)
        
        for shot in set_shots:
            modelSetShot: ModelSetShot
            moves = ShotUtils.AllMoves(shot, self.floatToIntFactor[0], self.floatToIntFactor[1])

            if len(moves) == maxMoves:
                modelSetShot = ModelSetShot(shotGroupToId[shot.shotGroup], moves)
            else:
                extra = []
                extraCount = maxMoves - len(moves)
                for x in range(0,extraCount):
                    extra.append(empty)

                modelSetShot = ModelSetShot(shotGroupToId[shot.shotGroup], moves + extra)
            
            result.append(modelSetShot)

        return result

    def _matchesToData(self, matches: List[Match]) -> List[ModelSetShot]:
        allShots = MatchUtils.AllShots(matches)
        return self._setshotsToData(allShots)
        
        
           
    def _predictModel(self, model:Any, shotMoves:List[List[List[int]]] ) -> int:
        
        #data = shotMoves[:100]
        data = np.array(shotMoves)
        data = np.reshape(data, (1, data.shape[0],data.shape[1]*data.shape[2]))
        resultCategories = model.predict(data)
        result = int(np.argmax(resultCategories))
        return result

    


