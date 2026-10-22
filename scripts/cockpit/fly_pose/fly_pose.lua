-- Fly Pose (GH-26): receive the fly's pose over UDP and turn the fly driver's bones each frame.
--
-- The sender is fly_driver/cockpit/pose_stream.py. A datagram is ASCII:
--   FLY1 <seq> wheel throttle brake head_yaw wing_l wing_r antenna_l antenna_r   (degrees)
--
-- Applied here: head yaw on DRIVER:RIG_Head, throttle on DRIVER:RIG_Foot_R, brake on
-- DRIVER:RIG_Foot_L. Not applied:
--   * wheel: the car's own steer.ksanim already turns the arm bones, and so the front legs,
--     with the car's real steering, which is the policy's steer;
--   * wings and antennae: the GH-25 build has no node of their own to turn (wings ride on
--     RIG_Cest with the thorax, antennae are inside DRIVER:HELMET). See docs/fly-driver.md.
--
-- CSP API used (acc-lua-sdk, lib_scene.lua and ac_matrices.d.lua): ac.findNodes('carRoot:0'),
-- :findNodes, :getTransformationRaw, :getParent, :getWorldTransformationRaw,
-- :storeCurrentTransformation, mat4x4.rotation, mat4x4:mul. UDP is LuaSocket, shipped with
-- CSP as require('shared/socket') (acc-lua-internal README, "lua-shared"). Posing driver
-- bones live is what CSP's own VR rig does (acc-lua-internal, lua-module/src/vr_rig_base.lua).

local socket = require('shared/socket')

local PORT = 9871          -- pose_stream.DEFAULT_PORT
local STALE_SECONDS = 1.0  -- no pose for this long: put the bones back

-- Calibration, checked in the game (docs/fly-driver.md, "What to check in the game"):
local HEAD_SIGN = -1       -- +yaw must turn the head to the driver's right
local PEDAL_SIGN = 1       -- +press must push the foot forward, onto the pedal
-- Ask AC to skip its own driver-model update for a frame, as CSP's VR rig does. Only turn
-- this on if the head and feet do not move with it off: it may also hold the steering arms.
local HOLD_DRIVER_MODEL = false

local FIELDS = { 'wheel', 'throttle', 'brake', 'head_yaw', 'wing_left', 'wing_right',
  'antenna_left', 'antenna_right' }

local udp = assert(socket.udp())
assert(udp:setsockname('0.0.0.0', PORT))
udp:settimeout(0)

local pose, lastSeq, packets, lastPacket = nil, nil, 0, -math.huge
local joints = nil  -- built once the car's driver is in the scene

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

local function findJoints()
  local carRoot = ac.findNodes('carRoot:0')
  local function joint(name, field, sign, axisOf)
    local node = carRoot:findNodes(name)
    if #node == 0 then return nil end
    node:storeCurrentTransformation()  -- restored if the app reloads
    return { node = node, base = node:getTransformationRaw():clone(), field = field,
      sign = sign, axisOf = axisOf }
  end
  local up = function (car) return car.up end
  local side = function (car) return car.side end
  local head = joint('DRIVER:RIG_Head', 'head_yaw', HEAD_SIGN, up)
  local throttle = joint('DRIVER:RIG_Foot_R', 'throttle', PEDAL_SIGN, side)
  local brake = joint('DRIVER:RIG_Foot_L', 'brake', PEDAL_SIGN, side)
  if not (head and throttle and brake) then return nil end
  return { head, throttle, brake }
end

-- Turn a joint by `angle` about a world axis through its own pivot: the rotation is
-- applied in the parent's frame after the stored local transform (row vectors), and the
-- position is kept.
local function apply(joint, angleDeg, car)
  local parentWorld = joint.node:getParent():getWorldTransformationRaw()
  local axis = parentWorld:inverse():transformVector(joint.axisOf(car)):normalize()
  local posed = joint.base:mul(mat4x4.rotation(math.rad(joint.sign * angleDeg), axis))
  posed.position:set(joint.base.position)
  joint.node:getTransformationRaw():set(posed)
end

local function restore()
  for _, joint in ipairs(joints) do joint.node:getTransformationRaw():set(joint.base) end
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
  local car = ac.getCar(0)
  if pose == nil or os.preciseClock() - lastPacket > STALE_SECONDS then
    if pose ~= nil then restore(); pose = nil end
    return
  end
  for _, joint in ipairs(joints) do apply(joint, pose[joint.field], car) end
  if HOLD_DRIVER_MODEL and ac.updateDriverModel then ac.updateDriverModel(0, true) end
end

function windowMain(dt)
  ui.text(string.format('UDP %d: %d poses', PORT, packets))
  ui.text(joints and 'fly bones found' or 'waiting for the driver model')
  if pose then
    ui.text(string.format('seq %d  head %+.1f  throttle %.1f  brake %.1f', lastSeq,
      pose.head_yaw, pose.throttle, pose.brake))
  else
    ui.text('no live pose: bones at rest')
  end
end
