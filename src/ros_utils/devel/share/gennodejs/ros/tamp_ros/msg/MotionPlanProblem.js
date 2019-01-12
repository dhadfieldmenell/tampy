// Auto-generated. Do not edit!

// (in-package tamp_ros.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let std_msgs = _finder('std_msgs');

//-----------------------------------------------------------

class MotionPlanProblem {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.solver_id = null;
      this.prob_id = null;
      this.server_id = null;
      this.task = null;
      this.state = null;
      this.cond = null;
      this.traj_mean = null;
      this.use_prior = null;
      this.sigma = null;
      this.mu = null;
      this.logmass = null;
      this.mass = null;
      this.N = null;
      this.K = null;
      this.Do = null;
    }
    else {
      if (initObj.hasOwnProperty('solver_id')) {
        this.solver_id = initObj.solver_id
      }
      else {
        this.solver_id = 0;
      }
      if (initObj.hasOwnProperty('prob_id')) {
        this.prob_id = initObj.prob_id
      }
      else {
        this.prob_id = 0;
      }
      if (initObj.hasOwnProperty('server_id')) {
        this.server_id = initObj.server_id
      }
      else {
        this.server_id = 0;
      }
      if (initObj.hasOwnProperty('task')) {
        this.task = initObj.task
      }
      else {
        this.task = '';
      }
      if (initObj.hasOwnProperty('state')) {
        this.state = initObj.state
      }
      else {
        this.state = [];
      }
      if (initObj.hasOwnProperty('cond')) {
        this.cond = initObj.cond
      }
      else {
        this.cond = 0;
      }
      if (initObj.hasOwnProperty('traj_mean')) {
        this.traj_mean = initObj.traj_mean
      }
      else {
        this.traj_mean = [];
      }
      if (initObj.hasOwnProperty('use_prior')) {
        this.use_prior = initObj.use_prior
      }
      else {
        this.use_prior = false;
      }
      if (initObj.hasOwnProperty('sigma')) {
        this.sigma = initObj.sigma
      }
      else {
        this.sigma = [];
      }
      if (initObj.hasOwnProperty('mu')) {
        this.mu = initObj.mu
      }
      else {
        this.mu = [];
      }
      if (initObj.hasOwnProperty('logmass')) {
        this.logmass = initObj.logmass
      }
      else {
        this.logmass = [];
      }
      if (initObj.hasOwnProperty('mass')) {
        this.mass = initObj.mass
      }
      else {
        this.mass = [];
      }
      if (initObj.hasOwnProperty('N')) {
        this.N = initObj.N
      }
      else {
        this.N = 0;
      }
      if (initObj.hasOwnProperty('K')) {
        this.K = initObj.K
      }
      else {
        this.K = 0;
      }
      if (initObj.hasOwnProperty('Do')) {
        this.Do = initObj.Do
      }
      else {
        this.Do = 0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type MotionPlanProblem
    // Serialize message field [solver_id]
    bufferOffset = _serializer.int32(obj.solver_id, buffer, bufferOffset);
    // Serialize message field [prob_id]
    bufferOffset = _serializer.int32(obj.prob_id, buffer, bufferOffset);
    // Serialize message field [server_id]
    bufferOffset = _serializer.int32(obj.server_id, buffer, bufferOffset);
    // Serialize message field [task]
    bufferOffset = _serializer.string(obj.task, buffer, bufferOffset);
    // Serialize message field [state]
    bufferOffset = _arraySerializer.float32(obj.state, buffer, bufferOffset, null);
    // Serialize message field [cond]
    bufferOffset = _serializer.int32(obj.cond, buffer, bufferOffset);
    // Serialize message field [traj_mean]
    // Serialize the length for message field [traj_mean]
    bufferOffset = _serializer.uint32(obj.traj_mean.length, buffer, bufferOffset);
    obj.traj_mean.forEach((val) => {
      bufferOffset = std_msgs.msg.Float32MultiArray.serialize(val, buffer, bufferOffset);
    });
    // Serialize message field [use_prior]
    bufferOffset = _serializer.bool(obj.use_prior, buffer, bufferOffset);
    // Serialize message field [sigma]
    bufferOffset = _arraySerializer.float32(obj.sigma, buffer, bufferOffset, null);
    // Serialize message field [mu]
    bufferOffset = _arraySerializer.float32(obj.mu, buffer, bufferOffset, null);
    // Serialize message field [logmass]
    bufferOffset = _arraySerializer.float32(obj.logmass, buffer, bufferOffset, null);
    // Serialize message field [mass]
    bufferOffset = _arraySerializer.float32(obj.mass, buffer, bufferOffset, null);
    // Serialize message field [N]
    bufferOffset = _serializer.int32(obj.N, buffer, bufferOffset);
    // Serialize message field [K]
    bufferOffset = _serializer.int32(obj.K, buffer, bufferOffset);
    // Serialize message field [Do]
    bufferOffset = _serializer.int32(obj.Do, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type MotionPlanProblem
    let len;
    let data = new MotionPlanProblem(null);
    // Deserialize message field [solver_id]
    data.solver_id = _deserializer.int32(buffer, bufferOffset);
    // Deserialize message field [prob_id]
    data.prob_id = _deserializer.int32(buffer, bufferOffset);
    // Deserialize message field [server_id]
    data.server_id = _deserializer.int32(buffer, bufferOffset);
    // Deserialize message field [task]
    data.task = _deserializer.string(buffer, bufferOffset);
    // Deserialize message field [state]
    data.state = _arrayDeserializer.float32(buffer, bufferOffset, null)
    // Deserialize message field [cond]
    data.cond = _deserializer.int32(buffer, bufferOffset);
    // Deserialize message field [traj_mean]
    // Deserialize array length for message field [traj_mean]
    len = _deserializer.uint32(buffer, bufferOffset);
    data.traj_mean = new Array(len);
    for (let i = 0; i < len; ++i) {
      data.traj_mean[i] = std_msgs.msg.Float32MultiArray.deserialize(buffer, bufferOffset)
    }
    // Deserialize message field [use_prior]
    data.use_prior = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [sigma]
    data.sigma = _arrayDeserializer.float32(buffer, bufferOffset, null)
    // Deserialize message field [mu]
    data.mu = _arrayDeserializer.float32(buffer, bufferOffset, null)
    // Deserialize message field [logmass]
    data.logmass = _arrayDeserializer.float32(buffer, bufferOffset, null)
    // Deserialize message field [mass]
    data.mass = _arrayDeserializer.float32(buffer, bufferOffset, null)
    // Deserialize message field [N]
    data.N = _deserializer.int32(buffer, bufferOffset);
    // Deserialize message field [K]
    data.K = _deserializer.int32(buffer, bufferOffset);
    // Deserialize message field [Do]
    data.Do = _deserializer.int32(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += object.task.length;
    length += 4 * object.state.length;
    object.traj_mean.forEach((val) => {
      length += std_msgs.msg.Float32MultiArray.getMessageSize(val);
    });
    length += 4 * object.sigma.length;
    length += 4 * object.mu.length;
    length += 4 * object.logmass.length;
    length += 4 * object.mass.length;
    return length + 57;
  }

  static datatype() {
    // Returns string type for a message object
    return 'tamp_ros/MotionPlanProblem';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '62f92843a78550529c22bfbaad9b886c';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    int32 solver_id
    int32 prob_id
    int32 server_id
    string task
    float32[] state
    int32 cond
    std_msgs/Float32MultiArray[] traj_mean
    
    bool use_prior
    float32[] sigma
    float32[] mu
    float32[] logmass
    float32[] mass
    int32 N
    int32 K
    int32 Do
    
    ================================================================================
    MSG: std_msgs/Float32MultiArray
    # Please look at the MultiArrayLayout message definition for
    # documentation on all multiarrays.
    
    MultiArrayLayout  layout        # specification of data layout
    float32[]         data          # array of data
    
    
    ================================================================================
    MSG: std_msgs/MultiArrayLayout
    # The multiarray declares a generic multi-dimensional array of a
    # particular data type.  Dimensions are ordered from outer most
    # to inner most.
    
    MultiArrayDimension[] dim # Array of dimension properties
    uint32 data_offset        # padding elements at front of data
    
    # Accessors should ALWAYS be written in terms of dimension stride
    # and specified outer-most dimension first.
    # 
    # multiarray(i,j,k) = data[data_offset + dim_stride[1]*i + dim_stride[2]*j + k]
    #
    # A standard, 3-channel 640x480 image with interleaved color channels
    # would be specified as:
    #
    # dim[0].label  = "height"
    # dim[0].size   = 480
    # dim[0].stride = 3*640*480 = 921600  (note dim[0] stride is just size of image)
    # dim[1].label  = "width"
    # dim[1].size   = 640
    # dim[1].stride = 3*640 = 1920
    # dim[2].label  = "channel"
    # dim[2].size   = 3
    # dim[2].stride = 3
    #
    # multiarray(i,j,k) refers to the ith row, jth column, and kth channel.
    
    ================================================================================
    MSG: std_msgs/MultiArrayDimension
    string label   # label of given dimension
    uint32 size    # size of given dimension (in type units)
    uint32 stride  # stride of given dimension
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new MotionPlanProblem(null);
    if (msg.solver_id !== undefined) {
      resolved.solver_id = msg.solver_id;
    }
    else {
      resolved.solver_id = 0
    }

    if (msg.prob_id !== undefined) {
      resolved.prob_id = msg.prob_id;
    }
    else {
      resolved.prob_id = 0
    }

    if (msg.server_id !== undefined) {
      resolved.server_id = msg.server_id;
    }
    else {
      resolved.server_id = 0
    }

    if (msg.task !== undefined) {
      resolved.task = msg.task;
    }
    else {
      resolved.task = ''
    }

    if (msg.state !== undefined) {
      resolved.state = msg.state;
    }
    else {
      resolved.state = []
    }

    if (msg.cond !== undefined) {
      resolved.cond = msg.cond;
    }
    else {
      resolved.cond = 0
    }

    if (msg.traj_mean !== undefined) {
      resolved.traj_mean = new Array(msg.traj_mean.length);
      for (let i = 0; i < resolved.traj_mean.length; ++i) {
        resolved.traj_mean[i] = std_msgs.msg.Float32MultiArray.Resolve(msg.traj_mean[i]);
      }
    }
    else {
      resolved.traj_mean = []
    }

    if (msg.use_prior !== undefined) {
      resolved.use_prior = msg.use_prior;
    }
    else {
      resolved.use_prior = false
    }

    if (msg.sigma !== undefined) {
      resolved.sigma = msg.sigma;
    }
    else {
      resolved.sigma = []
    }

    if (msg.mu !== undefined) {
      resolved.mu = msg.mu;
    }
    else {
      resolved.mu = []
    }

    if (msg.logmass !== undefined) {
      resolved.logmass = msg.logmass;
    }
    else {
      resolved.logmass = []
    }

    if (msg.mass !== undefined) {
      resolved.mass = msg.mass;
    }
    else {
      resolved.mass = []
    }

    if (msg.N !== undefined) {
      resolved.N = msg.N;
    }
    else {
      resolved.N = 0
    }

    if (msg.K !== undefined) {
      resolved.K = msg.K;
    }
    else {
      resolved.K = 0
    }

    if (msg.Do !== undefined) {
      resolved.Do = msg.Do;
    }
    else {
      resolved.Do = 0
    }

    return resolved;
    }
};

module.exports = MotionPlanProblem;
