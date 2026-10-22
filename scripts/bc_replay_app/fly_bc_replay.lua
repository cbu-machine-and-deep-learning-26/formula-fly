-- GH-29: replay stepper for behaviour-cloning frames. Copy this folder to
-- <AC>\apps\lua\fly_bc_replay. In a replay, it holds the playback at the instant
-- scripts/bc_replay_capture.py asks for and tells it when that instant is on screen.
--
-- Shared memory Local\fly_bc_replay, all 4-byte fields:
--   want     written by the capturer: replay step k + 1 to show (step = 20 ms); 0 = idle
--   shown    written here: equals want once that step has been on screen SETTLE_FRAMES frames
--   the rest written here every frame, for the capturer's checks and log
local mm = ac.writeMemoryMappedFile('fly_bc_replay', [[
  int want; int shown; int frames; int current; float frameMs; int replay;
  float x; float y; float z; int heartbeat; int camera; int drivable;
]])

local STEP_MS = 20
-- Frames rendered at the new instant before the capturer may grab it: the capture is a
-- screen grab, so the frame it sees must have been presented, not just simulated.
local SETTLE_FRAMES = 3
local target = 0
local settled = 0

function script.update(dt)
  local sim = ac.getSim()
  mm.heartbeat = mm.heartbeat + 1
  mm.replay = sim.isReplayActive and 1 or 0
  mm.frames = sim.replayFrames
  mm.current = sim.replayCurrentFrame
  mm.frameMs = sim.replayFrameMs
  mm.camera = sim.cameraMode
  mm.drivable = sim.driveableCameraMode
  local car = ac.getCar(0)
  if car then
    mm.x, mm.y, mm.z = car.position.x, car.position.y, car.position.z
  end
  if not sim.isReplayActive or sim.replayFrames < 2 then return end

  ac.focusCar(0)
  ac.setAppsHidden(true)
  if sim.cameraMode ~= ac.CameraMode.Drivable then ac.setCurrentCamera(ac.CameraMode.Drivable) end
  if sim.driveableCameraMode ~= ac.DrivableCamera.Dash then
    ac.setCurrentDrivableCamera(ac.DrivableCamera.Dash)
  end

  if mm.want <= 0 then return end
  local t = (mm.want - 1) * STEP_MS
  local frame = math.floor(t / sim.replayFrameMs)
  local blend = t / sim.replayFrameMs - frame
  if frame >= sim.replayFrames - 1 then frame, blend = sim.replayFrames - 2, 1 end
  -- Set every frame, not once: it also stops the replay from playing on.
  ac.setReplayPosition(frame, blend)

  if mm.want ~= target then
    target, settled, mm.shown = mm.want, 0, 0
  else
    settled = settled + 1
    if settled >= SETTLE_FRAMES then mm.shown = target end
  end
end
