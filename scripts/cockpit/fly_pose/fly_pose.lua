-- Fly Pose (GH-26): receive the fly's pose over UDP and turn the fly driver's bones each frame.
--
-- The sender is fly_driver/cockpit/pose_stream.py. A datagram is ASCII:
--   FLY1 <seq> wheel throttle brake head_yaw wing_l wing_r antenna_l antenna_r   (degrees)
--
-- Applied here:
--   * a fixed head pitch on DRIVER:RIG_Head, about the car's side axis, so the fly looks
--     ahead instead of up and back along its reclined body; applied every frame, pose or not;
--   * head yaw on DRIVER:RIG_Head, about the car's up axis;
--   * throttle on DRIVER:RIG_Foot_R and brake on DRIVER:RIG_Foot_L, about the car's side axis;
--   * wings on FLY:WING_L / FLY:WING_R and antennae on FLY:ANTENNA_L / FLY:ANTENNA_R, about
--     each node's own x axis, which the build puts along the flybody hinge joint
--     (fly_driver/cockpit/seated_fly.py, HINGE_NODES).
-- Not applied: the wheel. The car's own steer.ksanim already turns the arm bones, and so the
-- front legs, with the car's real steering, which is the policy's steer.
--
-- CSP API used (acc-lua-sdk, lib_scene.lua and ac_matrices.d.lua): ac.findNodes('carRoot:0'),
-- :findNodes, :getTransformationRaw, :getParent, :getWorldTransformationRaw, :setVisible,
-- :storeCurrentTransformation, mat4x4.rotation, mat4x4:mul. UDP is LuaSocket, shipped with
-- CSP as require('shared/socket') (acc-lua-internal README, "lua-shared"). Posing driver
-- bones live is what CSP's own VR rig does (acc-lua-internal, lua-module/src/vr_rig_base.lua).

local socket = require('shared/socket')

local PORT = 9871          -- pose_stream.DEFAULT_PORT
local STALE_SECONDS = 1.0  -- no pose for this long: back to the rest pose

-- Calibration, checked in the game (docs/fly-driver.md, "What to check in the game"):
local HEAD_SIGN = -1       -- +yaw must turn the head to the driver's right
local PEDAL_SIGN = 1       -- +press must push the foot forward, onto the pedal
local HINGE_SIGN = 1       -- +wing angle must spread the folded wings out from the body
-- The seated fly looks along its body, up and back. Turning the head this far about the car's
-- side axis levels its gaze: seated_fly.level_gaze_pitch_deg, which pose_fly_driver.py prints
-- for the car it posed (141 for the SF70H). It is past a real fly's neck (flybody's head joint
-- bends -29 to +17 degrees) on purpose; reclining the body less put the fly through the floor.
local HEAD_PITCH_DEG = 141
local HEAD_PITCH_SIGN = 1  -- +pitch must turn the face forward and down, towards the road ahead
-- Ask AC to skip its own driver-model update for a frame, as CSP's VR rig does. Only turn
-- this on if the head and feet do not move with it off: it may also hold the steering arms.
local HOLD_DRIVER_MODEL = false

local FIELDS = { 'wheel', 'throttle', 'brake', 'head_yaw', 'wing_left', 'wing_right',
  'antenna_left', 'antenna_right' }
-- live_pose.LIMITS_DEG: the sender clips already; this guards against any other sender.
local LIMITS = { throttle = { 0, 25 }, brake = { 0, 25 }, head_yaw = { -30, 30 },
  wing_left = { 0, 10 }, wing_right = { 0, 10 }, antenna_left = { -15, 15 },
  antenna_right = { -15, 15 } }

local udp = assert(socket.udp())
assert(udp:setsockname('0.0.0.0', PORT))
udp:settimeout(0)

local pose, lastSeq, packets, lastPacket = nil, nil, 0, -math.huge
local joints, antennae = nil, nil  -- found once the car's driver is in the scene

local function decode(datagram)
  local tokens = {}
  for token in datagram:gmatch('%S+') do tokens[#tokens + 1] = token end
  if tokens[1] ~= 'FLY1' or #tokens ~= 2 + #FIELDS then return nil end
  local values = {}
  for i, name in ipairs(FIELDS) do
    values[name] = tonumber(tokens[i + 2])
    if values[name] == nil then return nil end
  end
  return tonumber(tokens[2]), values
end

-- A joint turns about a world axis (`axisOf(car)`) through its pivot, or, with `axisOf`
-- nil, about its own local x.
local function findJoints()
  local carRoot = ac.findNodes('carRoot:0')
  local found = {}
  local function joint(name, field, sign, axisOf)
    local node = carRoot:findNodes(name)
    if #node == 0 then return nil end
    node:storeCurrentTransformation()  -- restored if the app reloads
    found[#found + 1] = { node = node, base = node:getTransformationRaw():clone(),
      field = field, sign = sign, axisOf = axisOf }
    return found[#found]
  end
  local up = function (car) return car.up end
  local side = function (car) return car.side end
  local head = joint('DRIVER:RIG_Head', 'head_yaw', HEAD_SIGN, up)
  if head then head.pitchDeg = HEAD_PITCH_SIGN * HEAD_PITCH_DEG end
  local ok = head
    and joint('DRIVER:RIG_Foot_R', 'throttle', PEDAL_SIGN, side)
    and joint('DRIVER:RIG_Foot_L', 'brake', PEDAL_SIGN, side)
    and joint('FLY:WING_L', 'wing_left', HINGE_SIGN, nil)
    and joint('FLY:WING_R', 'wing_right', HINGE_SIGN, nil)
    and joint('FLY:ANTENNA_L', 'antenna_left', 1, nil)
    and joint('FLY:ANTENNA_R', 'antenna_right', 1, nil)
  if not ok then return nil end
  return found
end

local function apply(joint, angleDeg, car)
  local limit = LIMITS[joint.field]
  local angle = math.rad(joint.sign * math.clamp(angleDeg, limit[1], limit[2]))
  local posed
  if joint.axisOf then
    -- In the parent's frame, after the stored local transform (row vectors); keep the pivot.
    local toParent = joint.node:getParent():getWorldTransformationRaw():inverse()
    local axis = toParent:transformVector(joint.axisOf(car)):normalize()
    posed = joint.base
    if joint.pitchDeg then
      local sideAxis = toParent:transformVector(car.side):normalize()
      posed = posed:mul(mat4x4.rotation(math.rad(joint.pitchDeg), sideAxis))
    end
    posed = posed:mul(mat4x4.rotation(angle, axis))
    posed.position:set(joint.base.position)
  else
    -- In the node's own frame, before the stored local transform: about its hinge.
    posed = mat4x4.rotation(angle, vec3(1, 0, 0)):mul(joint.base)
  end
  joint.node:getTransformationRaw():set(posed)
end

function script.update(dt)
  while true do
    local datagram = udp:receive()
    if not datagram then break end
    local seq, values = decode(datagram)
    if seq then pose, lastSeq, packets, lastPacket = values, seq, packets + 1, os.preciseClock() end
  end

  joints = joints or findJoints()
  if not joints then return end
  -- The antennae used to be inside DRIVER:HELMET, which cars hide in the cockpit camera.
  -- On their own nodes they need hiding here, or they sit in front of the lens.
  antennae = antennae or ac.findNodes('carRoot:0'):findNodes('{ FLY:ANTENNA_L, FLY:ANTENNA_R }')
  antennae:setVisible(ac.getSim().cameraMode ~= ac.CameraMode.Cockpit)

  local car = ac.getCar(0)
  if pose == nil or os.preciseClock() - lastPacket > STALE_SECONDS then
    -- Rest pose, with the head still sat up: every joint at 0, which is its base except the
    -- head's fixed pitch. Every frame, because the game resets the head bone from the .knh.
    pose = nil
    for _, joint in ipairs(joints) do apply(joint, 0, car) end
    return
  end
  for _, joint in ipairs(joints) do apply(joint, pose[joint.field], car) end
  if HOLD_DRIVER_MODEL and ac.updateDriverModel then ac.updateDriverModel(0, true) end
end

function windowMain(dt)
  ui.text(string.format('UDP %d: %d poses', PORT, packets))
  ui.text(joints and 'fly bones and hinges found' or 'waiting for the driver model')
  if pose then
    ui.text(string.format('seq %d  head %+.1f  throttle %.1f  brake %.1f', lastSeq,
      pose.head_yaw, pose.throttle, pose.brake))
    ui.text(string.format('wings %.1f %.1f  antennae %+.1f %+.1f', pose.wing_left,
      pose.wing_right, pose.antenna_left, pose.antenna_right))
  else
    ui.text('no live pose: bones at rest, head sat up')
  end
end
