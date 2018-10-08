;; Auto-generated. Do not edit!


(when (boundp 'tamp_ros::PolicyUpdate)
  (if (not (find-package "TAMP_ROS"))
    (make-package "TAMP_ROS"))
  (shadow 'PolicyUpdate (find-package "TAMP_ROS")))
(unless (find-package "TAMP_ROS::POLICYUPDATE")
  (make-package "TAMP_ROS::POLICYUPDATE"))

(in-package "ROS")
;;//! \htmlinclude PolicyUpdate.msg.html


(defclass tamp_ros::PolicyUpdate
  :super ros::object
  :slots (_obs _mu _prc _wt _dO _dPrimObs _dU _n _rollout_len ))

(defmethod tamp_ros::PolicyUpdate
  (:init
   (&key
    ((:obs __obs) (make-array 0 :initial-element 0.0 :element-type :float))
    ((:mu __mu) (make-array 0 :initial-element 0.0 :element-type :float))
    ((:prc __prc) (make-array 0 :initial-element 0.0 :element-type :float))
    ((:wt __wt) (make-array 0 :initial-element 0.0 :element-type :float))
    ((:dO __dO) 0)
    ((:dPrimObs __dPrimObs) 0)
    ((:dU __dU) 0)
    ((:n __n) 0)
    ((:rollout_len __rollout_len) 0)
    )
   (send-super :init)
   (setq _obs __obs)
   (setq _mu __mu)
   (setq _prc __prc)
   (setq _wt __wt)
   (setq _dO (round __dO))
   (setq _dPrimObs (round __dPrimObs))
   (setq _dU (round __dU))
   (setq _n (round __n))
   (setq _rollout_len (round __rollout_len))
   self)
  (:obs
   (&optional __obs)
   (if __obs (setq _obs __obs)) _obs)
  (:mu
   (&optional __mu)
   (if __mu (setq _mu __mu)) _mu)
  (:prc
   (&optional __prc)
   (if __prc (setq _prc __prc)) _prc)
  (:wt
   (&optional __wt)
   (if __wt (setq _wt __wt)) _wt)
  (:dO
   (&optional __dO)
   (if __dO (setq _dO __dO)) _dO)
  (:dPrimObs
   (&optional __dPrimObs)
   (if __dPrimObs (setq _dPrimObs __dPrimObs)) _dPrimObs)
  (:dU
   (&optional __dU)
   (if __dU (setq _dU __dU)) _dU)
  (:n
   (&optional __n)
   (if __n (setq _n __n)) _n)
  (:rollout_len
   (&optional __rollout_len)
   (if __rollout_len (setq _rollout_len __rollout_len)) _rollout_len)
  (:serialization-length
   ()
   (+
    ;; float32[] _obs
    (* 4    (length _obs)) 4
    ;; float32[] _mu
    (* 4    (length _mu)) 4
    ;; float32[] _prc
    (* 4    (length _prc)) 4
    ;; float32[] _wt
    (* 4    (length _wt)) 4
    ;; int32 _dO
    4
    ;; int32 _dPrimObs
    4
    ;; int32 _dU
    4
    ;; int32 _n
    4
    ;; int32 _rollout_len
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32[] _obs
     (write-long (length _obs) s)
     (dotimes (i (length _obs))
       (sys::poke (elt _obs i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;; float32[] _mu
     (write-long (length _mu) s)
     (dotimes (i (length _mu))
       (sys::poke (elt _mu i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;; float32[] _prc
     (write-long (length _prc) s)
     (dotimes (i (length _prc))
       (sys::poke (elt _prc i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;; float32[] _wt
     (write-long (length _wt) s)
     (dotimes (i (length _wt))
       (sys::poke (elt _wt i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;; int32 _dO
       (write-long _dO s)
     ;; int32 _dPrimObs
       (write-long _dPrimObs s)
     ;; int32 _dU
       (write-long _dU s)
     ;; int32 _n
       (write-long _n s)
     ;; int32 _rollout_len
       (write-long _rollout_len s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32[] _obs
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _obs (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _obs i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     ))
   ;; float32[] _mu
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _mu (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _mu i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     ))
   ;; float32[] _prc
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _prc (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _prc i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     ))
   ;; float32[] _wt
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _wt (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _wt i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     ))
   ;; int32 _dO
     (setq _dO (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; int32 _dPrimObs
     (setq _dPrimObs (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; int32 _dU
     (setq _dU (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; int32 _n
     (setq _n (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; int32 _rollout_len
     (setq _rollout_len (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get tamp_ros::PolicyUpdate :md5sum-) "74d9a9ad258b0d5854987033dafe686d")
(setf (get tamp_ros::PolicyUpdate :datatype-) "tamp_ros/PolicyUpdate")
(setf (get tamp_ros::PolicyUpdate :definition-)
      "float32[] obs
float32[] mu
float32[] prc
float32[] wt

int32 dO
int32 dPrimObs
int32 dU
int32 n
int32 rollout_len

")



(provide :tamp_ros/PolicyUpdate "74d9a9ad258b0d5854987033dafe686d")


