#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on February 20, 2026, at 14:45
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

import psychopy
psychopy.useVersion('2024.2.4')


# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '1'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.4'
expName = 'infant_attention'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': '',
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1536, 864]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\nylah\\InfantAttentionStudy\\infant_attention.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='event'
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='Pyglet',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='Pyglet'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "instructions" ---
    Instructions = visual.TextStim(win=win, name='Instructions',
        text='Press SPACE to Begin',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "attention_getter" ---
    background_image_attention_getter = visual.ImageStim(
        win=win,
        name='background_image_attention_getter', 
        image='Trees.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2, 1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    attention_getter_laugh = visual.MovieStim(
        win, name='attention_getter_laugh',
        filename='baby video.mp4', movieLib='ffpyplayer',
        loop=True, volume=3.0, noAudio=False,
        pos=(0, 0), size=(0.28, 0.28), units=win.units,
        ori=0.0, anchor='center',opacity=None, contrast=1.0,
        depth=-1
    )
    
    # --- Initialize components for Routine "trial" ---
    background_image = visual.ImageStim(
        win=win,
        name='background_image', 
        image='Trees.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2, 1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    face_stim = visual.ImageStim(
        win=win,
        name='face_stim', units='height', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=1.0,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    occluder_stim = visual.ImageStim(
        win=win,
        name='occluder_stim', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    # Run 'Begin Experiment' code from trial_code
    #Normalize the sound stream
    sound_stream = sound.Sound('sounds/Bah.wav')
    
    
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "instructions" ---
    # create an object to store info about Routine instructions
    instructions = data.Routine(
        name='instructions',
        components=[Instructions],
    )
    instructions.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for instructions
    instructions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions.tStart = globalClock.getTime(format='float')
    instructions.status = STARTED
    thisExp.addData('instructions.started', instructions.tStart)
    instructions.maxDuration = None
    # keep track of which components have finished
    instructionsComponents = instructions.components
    for thisComponent in instructions.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions" ---
    instructions.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Instructions* updates
        
        # if Instructions is starting this frame...
        if Instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Instructions.frameNStart = frameN  # exact frame index
            Instructions.tStart = t  # local t and not account for scr refresh
            Instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Instructions, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Instructions.started')
            # update status
            Instructions.status = STARTED
            Instructions.setAutoDraw(True)
        
        # if Instructions is active this frame...
        if Instructions.status == STARTED:
            # update params
            pass
        # Run 'Each Frame' code from instructions_code
        keys = event.getKeys()
        
        # Spacebar advances to next routine
        if 'space' in keys:
            print("Space pressed in instructions - continuing")
            continueRoutine = False
        
        # Escape quits experiment
        if 'escape' in keys:
            print("Escape pressed - quitting experiment")
            core.quit()
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instructions.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions" ---
    for thisComponent in instructions.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions
    instructions.tStop = globalClock.getTime(format='float')
    instructions.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions.stopped', instructions.tStop)
    thisExp.nextEntry()
    # the Routine "instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler2(
        name='trials',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('sample conditions.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial in trials:
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "attention_getter" ---
        # create an object to store info about Routine attention_getter
        attention_getter = data.Routine(
            name='attention_getter',
            components=[background_image_attention_getter, attention_getter_laugh],
        )
        attention_getter.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for attention_getter
        attention_getter.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        attention_getter.tStart = globalClock.getTime(format='float')
        attention_getter.status = STARTED
        thisExp.addData('attention_getter.started', attention_getter.tStart)
        attention_getter.maxDuration = None
        # keep track of which components have finished
        attention_getterComponents = attention_getter.components
        for thisComponent in attention_getter.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "attention_getter" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        attention_getter.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *background_image_attention_getter* updates
            
            # if background_image_attention_getter is starting this frame...
            if background_image_attention_getter.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                background_image_attention_getter.frameNStart = frameN  # exact frame index
                background_image_attention_getter.tStart = t  # local t and not account for scr refresh
                background_image_attention_getter.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(background_image_attention_getter, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'background_image_attention_getter.started')
                # update status
                background_image_attention_getter.status = STARTED
                background_image_attention_getter.setAutoDraw(True)
            
            # if background_image_attention_getter is active this frame...
            if background_image_attention_getter.status == STARTED:
                # update params
                pass
            
            # *attention_getter_laugh* updates
            
            # if attention_getter_laugh is starting this frame...
            if attention_getter_laugh.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                attention_getter_laugh.frameNStart = frameN  # exact frame index
                attention_getter_laugh.tStart = t  # local t and not account for scr refresh
                attention_getter_laugh.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(attention_getter_laugh, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'attention_getter_laugh.started')
                # update status
                attention_getter_laugh.status = STARTED
                attention_getter_laugh.setAutoDraw(True)
                attention_getter_laugh.play()
            
            # if attention_getter_laugh is stopping this frame...
            if attention_getter_laugh.status == STARTED:
                if bool(False) or attention_getter_laugh.isFinished:
                    # keep track of stop time/frame for later
                    attention_getter_laugh.tStop = t  # not accounting for scr refresh
                    attention_getter_laugh.tStopRefresh = tThisFlipGlobal  # on global time
                    attention_getter_laugh.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'attention_getter_laugh.stopped')
                    # update status
                    attention_getter_laugh.status = FINISHED
                    attention_getter_laugh.setAutoDraw(False)
                    attention_getter_laugh.stop()
            # Run 'Each Frame' code from attention_getter_code
            keys = event.getKeys()
            
            # Spacebar advances to trial
            if 'space' in keys:
                print("Space pressed in attention getter - continuing to trial")
                continueRoutine = False
            
            # Escape quits experiment
            if 'escape' in keys:
                print("Escape pressed - quitting experiment")
                core.quit()
            
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[attention_getter_laugh]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                attention_getter.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in attention_getter.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "attention_getter" ---
        for thisComponent in attention_getter.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for attention_getter
        attention_getter.tStop = globalClock.getTime(format='float')
        attention_getter.tStopRefresh = tThisFlipGlobal
        thisExp.addData('attention_getter.stopped', attention_getter.tStop)
        attention_getter_laugh.stop()  # ensure movie has stopped at end of Routine
        # the Routine "attention_getter" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "trial" ---
        # create an object to store info about Routine trial
        trial = data.Routine(
            name='trial',
            components=[background_image, face_stim, occluder_stim],
        )
        trial.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        face_stim.setImage(face)
        occluder_stim.setImage(occluder)
        # Run 'Begin Routine' code from trial_code
        # Load trial variables from Excel
        current_sequence = str(sequence)
        
        # Map A/B/C to their respective sound file paths
        sound_map = {}
        sound_map['A'] = str(sound_A)
        sound_map['B'] = str(sound_B)
        sound_map['C'] = str(sound_C)
        
        # Create two lists: one for sound objects, one for the corresponding labels
        files_to_play = []  # List of sound objects
        labels_to_play = []  # Corresponding label list for reference
        
        # Loop through sequence and create sound objects using psysound (safe import)
        for char in current_sequence:
            if char in sound_map:
                files_to_play.append(sound_map[char])
                labels_to_play.append(char)
                
        # Initialize control variables
        current_step = 0
        waiting_for_sound = True
        
        # Animation settings
        animation_phase = 0  # 0=waiting, 1=rising, 2=visible, 3=lowering
        animation_start_time = 0
        rise_duration = 1.0  # Time to rise up from behind occluder
        visible_duration = 0.5  # Time to stay visible
        lower_duration = 1.0  # Time to lower back behind occluder
        
        # Position parameters
        start_y = 0  # Starting position (behind occluder)
        peak_y = 0.25     # Peak position (visible above occluder)
        
        # Adjust sizes and initial positions
        face_stim.size = (0.22, 0.22)       # Size of face
        face_stim.pos = [0, start_y]      # Position face behind occluder
        occluder_stim.size = (0.23, 0.23) # Size of occluder
        occluder_stim.pos = [0, 0]     # Position of occluder
        
        space_held = False
        space_start_time = None
        
        # Set up keyboard state handler
        from pyglet.window import key
        keyboard_state = key.KeyStateHandler()
        win.winHandle.push_handlers(keyboard_state)
        # store start times for trial
        trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        trial.tStart = globalClock.getTime(format='float')
        trial.status = STARTED
        thisExp.addData('trial.started', trial.tStart)
        trial.maxDuration = None
        # keep track of which components have finished
        trialComponents = trial.components
        for thisComponent in trial.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trial" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        trial.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *background_image* updates
            
            # if background_image is starting this frame...
            if background_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                background_image.frameNStart = frameN  # exact frame index
                background_image.tStart = t  # local t and not account for scr refresh
                background_image.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(background_image, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'background_image.started')
                # update status
                background_image.status = STARTED
                background_image.setAutoDraw(True)
            
            # if background_image is active this frame...
            if background_image.status == STARTED:
                # update params
                pass
            
            # *face_stim* updates
            
            # if face_stim is starting this frame...
            if face_stim.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                face_stim.frameNStart = frameN  # exact frame index
                face_stim.tStart = t  # local t and not account for scr refresh
                face_stim.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(face_stim, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'face_stim.started')
                # update status
                face_stim.status = STARTED
                face_stim.setAutoDraw(True)
            
            # if face_stim is active this frame...
            if face_stim.status == STARTED:
                # update params
                pass
            
            # *occluder_stim* updates
            
            # if occluder_stim is starting this frame...
            if occluder_stim.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                occluder_stim.frameNStart = frameN  # exact frame index
                occluder_stim.tStart = t  # local t and not account for scr refresh
                occluder_stim.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(occluder_stim, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'occluder_stim.started')
                # update status
                occluder_stim.status = STARTED
                occluder_stim.setAutoDraw(True)
            
            # if occluder_stim is active this frame...
            if occluder_stim.status == STARTED:
                # update params
                pass
            # Run 'Each Frame' code from trial_code
            # Escape
            if event.getKeys(['escape']):
                core.quit()
                
            # Check if space is currently pressed
            space_is_down = keyboard_state[key.SPACE]
            
            # Detect press
            if space_is_down and not space_held:
                space_held = True
                space_start_time = t
                print(f"Space pressed at {t}")
            
            # Detect release
            if not space_is_down and space_held:
                looking_time = t - space_start_time
                thisExp.addData('looking_time', looking_time)
                print(f"Looking time: {looking_time}s")
                continueRoutine = False
            
            # Play sounds and animate face based on current phase
            if current_step < len(files_to_play):
                
                # Phase 0: Waiting to start next sound/animation
                if animation_phase == 0:
                    # Start the sound and begin rising animation
                    sound_stream.setSound(files_to_play[current_step])
                    sound_stream.play()
                    print(f"Playing sound {current_step + 1}: {files_to_play[current_step]} (label: {labels_to_play[current_step]})")
                    
                    # Start rising animation
                    animation_phase = 1
                    animation_start_time = t
                
                # Phase 1: Face rising from behind occluder
                elif animation_phase == 1:
                    # Calculate progress for rising (0-1)
                    elapsed = t - animation_start_time
                    progress = min(1.0, elapsed / rise_duration)
                    
                    # Apply easing for smoother animation
                    eased_progress = progress * progress * (3 - 2 * progress)
                    
                    # Calculate and set face position
                    current_y = start_y + eased_progress * (peak_y - start_y)
                    face_stim.pos = [0, current_y]
                    
                    # Check if rising is complete
                    if progress >= 1.0:
                        # Start visible phase
                        animation_phase = 2
                        animation_start_time = t
                
                # Phase 2: Face visible at peak position
                elif animation_phase == 2:
                    # Keep face at peak position
                    face_stim.pos = [0, peak_y]
                    
                    # Check if visible duration is complete
                    elapsed = t - animation_start_time
                    if elapsed >= visible_duration:
                        # Start lowering phase
                        animation_phase = 3
                        animation_start_time = t
                        
                # Phase 3: Face lowering back behind occluder
                elif animation_phase == 3:
                    # Calculate progress for lowering (0-1)
                    elapsed = t - animation_start_time
                    progress = min(1.0, elapsed / lower_duration)
                
                    # Apply easing for smoother animation
                    eased_progress = progress * progress * (3 - 2 * progress)
                
                    # Calculate and set face position
                    current_y = peak_y - eased_progress * (peak_y - start_y)
                    face_stim.pos = [0, current_y]
                
                    # Check if lowering is complete
                    if progress >= 1.0:
                        # Animation complete, wait for sound to finish
                        face_stim.pos = [0, start_y]  # Keep face at lowest position
                    
                        # Stay in phase 3 until sound finishes
                        if not sound_stream.isPlaying:
                            # Sound finished, move to next sound and reset animation
                            animation_phase = 0
                            current_step += 1
                            print(f"Sound playing: {sound_stream.isPlaying}, Animation phase: {animation_phase}, Current step: {current_step}")
                            
            else:
                # No more sounds to play — end the trial
                continueRoutine = False
            
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                trial.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trial.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial" ---
        for thisComponent in trial.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for trial
        trial.tStop = globalClock.getTime(format='float')
        trial.tStopRefresh = tThisFlipGlobal
        thisExp.addData('trial.stopped', trial.tStop)
        # Run 'End Routine' code from trial_code
        # Stop the reusable sound stream if it's still playing
        if sound_stream.isPlaying:
            sound_stream.stop()
        
        # the Routine "trial" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'trials'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
