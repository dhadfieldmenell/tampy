int32 solver_id
int32 prob_id
string server_id
string alg_id
string task
float32[] state
float32[] targets
int32 cond
std_msgs/Float32MultiArray[] traj_mean
std_msgs/Float32MultiArray[] obs
std_msgs/Float32MultiArray[] U

float32[] pol_K
float32[] pol_k
float32[] pol_S
float32[] chol
int32 T
int32 dU
int32 dX

bool use_prior
float32[] sigma
float32[] mu
float32[] logmass
float32[] mass
int32 N
int32 K
int32 Do
