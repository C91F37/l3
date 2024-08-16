using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class Byte : CogsAgent
{
    // Cache to speed up
    private Rigidbody rBody;
    private Timer gameTimer;
    private Transform baseTransform;
    private Vector3 startingPosition;
    private List<GameObject> allTargets = new List<GameObject>();

    private bool wasFrozenLastFrame;

    private const float REWARD_LASER_HIT = 0.5f;
    private const float PENALTY_LASER_MISS = -0.05f;
    private const float PENALTY_LASER_USE = -0.02f;
    private const float REWARD_FOR_TARGET_PICKUP = 0.2f;
    private const float PENALTY_FOR_BEING_FROZEN = -0.5f;
    private const float SPEED_REWARD_MULTIPLIER = 0.01f;
    private const float CARRYING_PENALTY_MULTIPLIER = -0.02f;
    private const float OPTIMAL_SPEED_RATIO = 0.8f; 

    // ------------------BASIC MONOBEHAVIOR FUNCTIONS-------------------
    
    // Initialize values
    protected override void Start()
    {
        base.Start();
        rBody = GetComponent<Rigidbody>();
        gameTimer = timer.GetComponent<Timer>();
        baseTransform = baseLocation;
        startingPosition = transform.localPosition;
        wasFrozenLastFrame = IsFrozen();
        AssignBasicRewards();
    }

    // For actual actions in the environment (e.g. movement, shoot laser)
    // that is done continuously
    protected override void FixedUpdate() {
        base.FixedUpdate();
        
        // //check for transition from not frozen -> frozen
        if (!wasFrozenLastFrame && IsFrozen())
        {
            AddReward(PENALTY_FOR_BEING_FROZEN);
        }
        wasFrozenLastFrame = IsFrozen();

        // OptimalSpeedRewards();
        LaserControl();

        // EnhancedLaserControl();
        
        // Movement based on DirToGo and RotateDir
        moveAgent(dirToGo, rotateDir);
    }
    
    // --------------------AGENT FUNCTIONS-------------------------


    private void OptimalSpeedRewards()
    {
        float optimalSpeed = 1.5f - (0.05f * carriedTargets.Count);
        float currentSpeed = rBody.velocity.magnitude;
        float efficiency = currentSpeed / optimalSpeed;
        if (efficiency >= OPTIMAL_SPEED_RATIO)
        {
            // reward speeding
            AddReward(SPEED_REWARD_MULTIPLIER * (1 - efficiency));
        }
        else 
        {   
            // penalize moving too slow
            AddReward(CARRYING_PENALTY_MULTIPLIER * (1 - efficiency));
        }

        // penalize carry
        AddReward(CARRYING_PENALTY_MULTIPLIER * carriedTargets.Count);
    }

    protected bool EnhancedLaserControl()
    {
        //call base to maintain original behaviour
        bool baseResult = base.LaserControl();
        if (baseResult) 
        {
            AddReward(PENALTY_LASER_USE);
        }

        //custom logic for Byte
        bool laserHit = false;

        if (IsLaserOn() && !IsFrozen())
        {   
            RaycastHit hit;
            Vector3 rayDir = 20f * transform.forward;
            if (Physics.SphereCast(transform.position, 0.25f, rayDir, out hit, 
                20f, LayerMask.GetMask("Default"), QueryTriggerInteraction.Ignore))
                {
                    if (hit.collider.gameObject.CompareTag("Player"))
                    {
                        laserHit = true;
                        AddReward(REWARD_LASER_HIT);
                    }
                }
            if (!laserHit)
            {
                AddReward(PENALTY_LASER_MISS);
            }
        }

        return baseResult;
    }

    // Get relevant information from the environment to effectively learn behavior
    public override void CollectObservations(VectorSensor sensor)
    {
        // Agent velocity in x and z axis 
        Vector3 localVelocity = transform.InverseTransformDirection(rBody.velocity);
        sensor.AddObservation(localVelocity.x);
        sensor.AddObservation(localVelocity.z);

        // Time remaning
        sensor.AddObservation(gameTimer.GetTimeRemaning());  

        // Agent's current rotation
        // var localRotation = transform.rotation;
        // sensor.AddObservation(transform.rotation.y);
        sensor.AddObservation(transform.localEulerAngles.y);

        // Agent and home base's position
        sensor.AddObservation((baseTransform.localPosition - transform.localPosition).normalized);
        sensor.AddObservation(Vector3.Distance(baseTransform.localPosition, transform.localPosition));

        // for each target in the environment, add: its position, whether it is being carried,
        // and whether it is in a base
        foreach (GameObject target in allTargets){
            Vector3 relativePosition = target.transform.localPosition - transform.localPosition;
            sensor.AddObservation(relativePosition.normalized); // Direction to target
            sensor.AddObservation(relativePosition.magnitude); // Distance to target
            sensor.AddObservation(target.GetComponent<Target>().GetCarried());
            sensor.AddObservation(target.GetComponent<Target>().GetInBase());
        }
        
        // Whether the agent is frozen
        sensor.AddObservation(IsFrozen());
    }

    // For manual override of controls. This function will use keyboard presses to simulate output from your NN 
    public override void Heuristic(in ActionBuffers actionsOut)
{
        var discreteActionsOut = actionsOut.DiscreteActions;
        discreteActionsOut[0] = 0; //Simulated NN output 0
        discreteActionsOut[1] = 0; //....................1
        discreteActionsOut[2] = 0; //....................2
        discreteActionsOut[3] = 0; //....................3

        //TODO-2: Uncomment this next line when implementing GoBackToBase();
        discreteActionsOut[4] = 0;

       
        if (Input.GetKey(KeyCode.I))
        {
            discreteActionsOut[0] = 1;
        }       
        if (Input.GetKey(KeyCode.K))
        {
            discreteActionsOut[0] = 2;
        }
        if (Input.GetKey(KeyCode.L))
        {
            discreteActionsOut[1] = 1;
        }
        if (Input.GetKey(KeyCode.J))
        {
            //TODO-1: Using the above as examples, set the action out for the left arrow press
            discreteActionsOut[1] = 2;
            
        }
        

        //Shoot
        if (Input.GetKey(KeyCode.Space)){
            discreteActionsOut[2] = 1;
        }

        //GoToNearestTarget
        if (Input.GetKey(KeyCode.A)){
            discreteActionsOut[3] = 1;
        }

        //TODO-2: S for the output for GoBackToBase();
        if (Input.GetKey(KeyCode.S)){
            discreteActionsOut[4] = 1;
        }
    }

        // What to do when an action is received (i.e. when the Brain gives the agent information about possible actions)
        public override void OnActionReceived(ActionBuffers actions){

        int forwardAxis = (int)actions.DiscreteActions[0]; //NN output 0

        //TODO-1: Set these variables to their appopriate item from the act list
        int rotateAxis = (int)actions.DiscreteActions[1]; //NN output 1
        int shootAxis = (int)actions.DiscreteActions[2]; //NN output 2
        int goToTargetAxis = (int)actions.DiscreteActions[3]; //NN output 3

        
        
        //TODO-2: Uncomment this next line and set it to the appropriate item from the act list
        int goToBaseAxis = (int)actions.DiscreteActions[4]; //NN output 4

        //TODO-2: Make sure to remember to add goToBaseAxis when working on that part!
        
        MovePlayer(forwardAxis, rotateAxis, shootAxis, goToTargetAxis, goToBaseAxis);

    }


// ----------------------ONTRIGGER AND ONCOLLISION FUNCTIONS------------------------
    // Called when object collides with or trigger (similar to collide but without physics) other objects
    protected override void OnTriggerEnter(Collider collision)
    {
        base.OnTriggerEnter(collision);

        if (collision.gameObject.CompareTag("HomeBase") && 
        collision.gameObject.GetComponent<HomeBase>().team == GetTeam())
        {
            //base return reward, need adjustment
            AddReward(0.8f + 1f * carriedTargets.Count);
        }
    }

    protected override void OnCollisionEnter(Collision collision) 
    {
        base.OnCollisionEnter(collision);

        //target is not in my base and is not being carried and I am not frozen
        if (collision.gameObject.CompareTag("Target") && 
            collision.gameObject.GetComponent<Target>().GetInBase() != GetTeam() && 
            collision.gameObject.GetComponent<Target>().GetCarried() == 0 && !IsFrozen())
        {
            //Add rewards here
            AddReward(REWARD_FOR_TARGET_PICKUP); 
        }

        if (collision.gameObject.CompareTag("Wall"))
        {
            AddReward(-0.75f);
        }

        if (collision.gameObject.CompareTag("Player"))
        {
            AddReward(-0.2f);
        }

    }



    //  --------------------------HELPERS---------------------------- 
     private void AssignBasicRewards() {
        rewardDict = new Dictionary<string, float>();

        rewardDict.Add("frozen", 0f);
        rewardDict.Add("shooting-laser", 0f);
        rewardDict.Add("hit-enemy", 0f);
        rewardDict.Add("dropped-one-target", 0f);
        rewardDict.Add("dropped-targets", 0f);
    }
    
    private void MovePlayer(int forwardAxis, int rotateAxis, int shootAxis, int goToTargetAxis, int goToBase)
    //TODO-2: Add goToBase as an argument to this function ^
    {
        dirToGo = Vector3.zero;
        rotateDir = Vector3.zero;

        Vector3 forward = transform.forward;
        Vector3 backward = -transform.forward;
        Vector3 right = transform.up;
        Vector3 left = -transform.up;

        //fowardAxis: 
            // 0 -> do nothing
            // 1 -> go forward
            // 2 -> go backward
        if (forwardAxis == 0){
            //do nothing. This case is not necessary to include, it's only here to explicitly show what happens in case 0
        }
        else if (forwardAxis == 1){
            dirToGo = forward;
        }
        else if (forwardAxis == 2){
            //TODO-1: Tell your agent to go backward!
            dirToGo = backward;

            
        }

        //rotateAxis: 
            // 0 -> do nothing
            // 1 -> go right
            // 2 -> go left
        if (rotateAxis == 0){
            //do nothing
        }
        
        //TODO-1 : Implement the other cases for rotateDir
        else if (rotateAxis == 1){
            rotateDir = right;
        }
        else if (rotateAxis == 2){
            rotateDir = left;
        }

        //shoot
        if (shootAxis == 1){
            SetLaser(true);
        }
        else {
            SetLaser(false);
        }

        //go to the nearest target
        if (goToTargetAxis == 1){
            GoToNearestTarget();
        }

        //TODO-2: Implement the case for goToBaseAxis
        if (goToBase == 1) {
            GoToBase();
        }
        
    }

    // Go to home base
    private void GoToBase(){
        TurnAndGo(GetYAngle(myBase));
    }

    // Go to the nearest target
    private void GoToNearestTarget(){
        GameObject target = GetNearestTarget();
        if (target != null){
            float rotation = GetYAngle(target);
            TurnAndGo(rotation);
        }        
    }

    // Rotate and go in specified direction
    private void TurnAndGo(float rotation){

        if(rotation < -5f){
            rotateDir = transform.up;
        }
        else if (rotation > 5f){
            rotateDir = -transform.up;
        }
        else {
            dirToGo = transform.forward;
        }
    }

    // return reference to nearest target
    protected GameObject GetNearestTarget(){
        float distance = 200;
        GameObject nearestTarget = null;
        foreach (var target in targets)
        {
            float currentDistance = Vector3.Distance(target.transform.localPosition, transform.localPosition);
            if (currentDistance < distance && target.GetComponent<Target>().GetCarried() == 0 && target.GetComponent<Target>().GetInBase() != team){
                distance = currentDistance;
                nearestTarget = target;
            }
        }
        return nearestTarget;
    }

    private float GetYAngle(GameObject target) {
        
       Vector3 targetDir = target.transform.position - transform.position;
       Vector3 forward = transform.forward;

      float angle = Vector3.SignedAngle(targetDir, forward, Vector3.up);
      return angle; 
        
    }
}