;; Auto-generated. Do not edit!


(when (boundp 'tamp_ros::PlanResult)
  (if (not (find-package "TAMP_ROS"))
    (make-package "TAMP_ROS"))
  (shadow 'PlanResult (find-package "TAMP_ROS")))
(unless (find-package "TAMP_ROS::PLANRESULT")
  (make-package "TAMP_ROS::PLANRESULT"))

(in-package "ROS")
;;//! \htmlinclude PlanResult.msg.html


(defclass tamp_ros::PlanResult
  :super ros::object
  :slots (_prob_id _trajectory _success _failed_preds ))

(defmethod tamp_ros::PlanResult
  (:init
   (&key
    ((:prob_id __prob_id) 0)
    ((:trajectory __trajectory) (let (r) (dotimes (i 0) (push (instance tamp_ros::FloatArray :init) r)) r))
    ((:success __success) nil)
    ((:failed_preds __failed_preds) (let (r) (dotimes (i 0) (push "" r)) r))
    )
   (send-super :init)
   (setq _prob_id (round __prob_id))
   (setq _trajectory __trajectory)
   (setq _success __success)
   (setq _failed_preds __failed_preds)
   self)
  (:prob_id
   (&optional __prob_id)
   (if __prob_id (setq _prob_id __prob_id)) _prob_id)
  (:trajectory
   (&rest __trajectory)
   (if (keywordp (car __trajectory))
       (send* _trajectory __trajectory)
     (progn
       (if __trajectory (setq _trajectory (car __trajectory)))
       _trajectory)))
  (:success
   (&optional __success)
   (if __success (setq _success __success)) _success)
  (:failed_preds
   (&optional __failed_preds)
   (if __failed_preds (setq _failed_preds __failed_preds)) _failed_preds)
  (:serialization-length
   ()
   (+
    ;; int64 _prob_id
    8
    ;; tamp_ros/FloatArray[] _trajectory
    (apply #'+ (send-all _trajectory :serialization-length)) 4
    ;; bool _success
    1
    ;; string[] _failed_preds
    (apply #'+ (mapcar #'(lambda (x) (+ 4 (length x))) _failed_preds)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; int64 _prob_id
#+(or :alpha :irix6 :x86_64)
       (progn (sys::poke _prob_id (send s :buffer) (send s :count) :long) (incf (stream-count s) 8))
#-(or :alpha :irix6 :x86_64)
       (cond ((and (class _prob_id) (= (length (_prob_id . bv)) 2)) ;; bignum
              (write-long (ash (elt (_prob_id . bv) 0) 0) s)
              (write-long (ash (elt (_prob_id . bv) 1) -1) s))
             ((and (class _prob_id) (= (length (_prob_id . bv)) 1)) ;; big1
              (write-long (elt (_prob_id . bv) 0) s)
              (write-long (if (>= _prob_id 0) 0 #xffffffff) s))
             (t                                         ;; integer
              (write-long _prob_id s)(write-long (if (>= _prob_id 0) 0 #xffffffff) s)))
     ;; tamp_ros/FloatArray[] _trajectory
     (write-long (length _trajectory) s)
     (dolist (elem _trajectory)
       (send elem :serialize s)
       )
     ;; bool _success
       (if _success (write-byte -1 s) (write-byte 0 s))
     ;; string[] _failed_preds
     (write-long (length _failed_preds) s)
     (dolist (elem _failed_preds)
       (write-long (length elem) s) (princ elem s)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; int64 _prob_id
#+(or :alpha :irix6 :x86_64)
      (setf _prob_id (prog1 (sys::peek buf ptr- :long) (incf ptr- 8)))
#-(or :alpha :irix6 :x86_64)
      (setf _prob_id (let ((b0 (prog1 (sys::peek buf ptr- :integer) (incf ptr- 4)))
                  (b1 (prog1 (sys::peek buf ptr- :integer) (incf ptr- 4))))
              (cond ((= b1 -1) b0)
                     ((and (= b1  0)
                           (<= lisp::most-negative-fixnum b0 lisp::most-positive-fixnum))
                      b0)
                    ((= b1  0) (make-instance bignum :size 1 :bv (integer-vector b0)))
                    (t (make-instance bignum :size 2 :bv (integer-vector b0 (ash b1 1)))))))
   ;; tamp_ros/FloatArray[] _trajectory
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _trajectory (let (r) (dotimes (i n) (push (instance tamp_ros::FloatArray :init) r)) r))
     (dolist (elem- _trajectory)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;; bool _success
     (setq _success (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; string[] _failed_preds
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _failed_preds (make-list n))
     (dotimes (i n)
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setf (elt _failed_preds i) (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
     ))
   ;;
   self)
  )

(setf (get tamp_ros::PlanResult :md5sum-) "d4f5f1c50852a30db764ffda62f46133")
(setf (get tamp_ros::PlanResult :datatype-) "tamp_ros/PlanResult")
(setf (get tamp_ros::PlanResult :definition-)
      "int64 prob_id
FloatArray[] trajectory
bool success
string[] failed_preds


================================================================================
MSG: tamp_ros/FloatArray
float32[] data

")



(provide :tamp_ros/PlanResult "d4f5f1c50852a30db764ffda62f46133")


