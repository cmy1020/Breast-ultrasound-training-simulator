# Probe-tissue simulation
<div align="center">
  <img width="900" height="360" src="img/image-probe-breast.png" alt="probe-breast interaction">
</div>

This project allows to simulate the deformation induced by an ultrasound (US) probe to soft tissues with the finite element method, using [SOFA Framework](https://www.sofa-framework.org/). 

This branch represents an upgrade of the original repository (in the *sofapython2* branch), in order to be used with the latest SOFA versions and SofaPython3. At the moment, only the constraint-based approach (LM approach) has been implemented with SofaPython3.
> Please keep in mind that results reported in our [paper](https://link.springer.com/article/10.1007/s11548-020-02183-2) were obtained with the old SofaPython plugin using the code in the sofapython2 branch. Therefore, it is possible that there are slight differences in performances if you try to reproduce those results with the SofaPython3 version.

For further details on the modelling strategies and a more in-depth analysis please refer to the paper.

* [Setup](#setup)
* [Structure](#structure)
* [Usage](#usage)
* [Implementation details](#implementation-details)
* [References](#references)

## Setup
Simulations are implemented using the SOFA Framework with the SofaPython3 plugin. 
Follow installation instructions for:
- [SOFA](https://www.sofa-framework.org/community/doc/getting-started/build/linux/)
- [SofaPython3](https://sofapython3.readthedocs.io/en/latest/menu/Compilation.html)

Please install required python modules by running `pip install -r requirements.txt`

Tested with: 
- SOFA 21.12 and python 3.9 on a system with Ubuntu 18.04. 
- SOFA 22.06 and python 3.8 on a system with Ubuntu 20.04.

## Structure
This repository contains:
- a main `simulation.py` file where the simulation scene is defined
- `objects/` directory, which contains the implementation of the main simulated objects (i.e., the breast and the probe)
- `components/` directory, which contains functions to wrap general SOFA components and are independent from the specific simulated objects in this repository

## Usage
To start the simulation, run: 
```
python simulation.py
```
from the repository directory. `input_parameters.yml` is a file containing all the main configuration parameters for the scene. You can decide which approach to use for your simulation by changing the value of the *type* parameter (NB: only "LM" is supported so far). Moreover, you can specify the main simulation parameters, model, paths to files.

## Implementation details
This section provides some more technical details on the setup of simulations using the different methods, together with some hints and tricks.

### Simulation setup for collision-based methods
All collision-based methods (*Penalty* and *LM*) are characterized by 2 main phases: collision detection and collision response. *Penalty* and *LM* differ for the way they handle the response to a collision event, but collision detection is performed in the same way.

#### A) Collision detection
Collision detection pipeline is set-up at the beginning of the scene and it is characterized by some main components:
1. *DefaultPipeline*: sets up the default pipeline for collision detection
2. *BruteForceBroadPhase*: defines the first strategy for detecting colliding objects (called "broad phase" collision detection); if the bounding boxes of two objects intersect, finer collision detection methods are activated 
3. *BVHNarrowPhase*: defines the algorithm to be used as soon as a pair of colliding models are detected by the broad phase component. It handles what is called "narrow phase" collision detection. This specific component relies on Bounding Volume Hierarchy (BVH) to discard all the non-intersecting elements of the two colliding objects. This component relies on an intersection detection component to identify intersecting areas.
4. A proximity method (*MinProximityIntersection* or *LocalMinDistance*): defines the strategy used to detect the precise collision area, if any, between possibly colliding pairs; it requires setting two important parameters: alarm and contact distances
5. *DefaultContactManager*: defines the collision response type (default, friction, sticky, etc.)

In order to define the objects in your scene which are involved in the collision problem, they have to be associated with collision models. Collision models are defined by the components *TriangleCollisionModel*, *LineCollisionModel* and *PointCollisionModel*: they all have to be specified in order to detect collisions in case triangles, lines and/or points of your meshes are intersecting. However, you can also use just one or a combination of these models to limit computation time, but at the sake of accuracy.

##### Best practices in the setup of contact problems
Simulations involving collisions introduce a higher level of complexity to your scene, usually requiring some initial parameter tuning in order to obtain the expected behavior and avoid instability. Below are some best practices to consider when you set-up your collision problem.
- *alarm distance* and *contact distance*: correctly tuning these parameters is critical for successful collision detection. *alarm distance* represents the distance between two points below which the corresponding objects are considered as colliding; *contact distance* represents the distance below which collision response is applied to the colliding objects. These two parameters must be expressed with the same unit of measure of your scene, and *alarm distance* must be greater than *contact distance*. Their values have to be tuned coherently with the maximum relative motion between the objects within a time step (e.g. if your objects move of 1cm in a time step and your *alarm distance* is 1mm, the collision will not be detected).
- Time step *dt*: when choosing your simulation *dt*, remember that its value will impact your collision distances and the amount of motion of your objects within a *dt*. In general, *dt* is the first parameter you should try to adjust (by decreasing it) in case your simulation turns out to be unstable.
- Delta motion within a time step: in order to properly detect collision pairs, it is important to check that the increment of motion for the two objects within a *dt* does not exceed the alarm distance. This increment motion depends on the chosen time step and the method used to define the movement (in our case, the *LinearMovementConstraint*)
- Initial not-contact condition: to guarantee proper behavior of collision detection algorithms, it is necessary that there is no initial interpenetration between the colliding objects. This means that the two objects should be at an initial distance greater than the alarm distance.
- Lower resolution for the collision meshes: SOFA allows you to specify a dedicated mesh for collision detection and response. If you are not interested in having a highly accurate behavior in the contact area (but you are more interested in the global deformed shape), it is advisable to use a lower resolution mesh for collision models. This will save computation time, since it limits the number of points/lines/triangles considered by the algorithm during collision checking. Collision mesh can also be a subpart of the complete mesh, if you already know which part of your object will collide (as is the case of the US probe in this repo). In general, collision meshes of different colliding objects should have comparable elements sizes.
- Mesh normals: collision is detected in the direction of normals, so check that your mesh normals are properly set.

<div align="center">
<img width="280" height="200" src="img/interaction_forces.png"/> 

Figure shows the interactions created between the rigid probe and the tissues whenever *contact distance* is overcome.
</div>

#### B) Collision response
##### Penalty method
Collision response is obtained by applying forces depending on the amount of interpenetration between the objects, through a proportionality constant called *contactStiffness*, which is found within the collision components. Fine tuning of *contactStiffness* value is needed to obtain a stable collision response, and it is problem dependent. There are two flags to set within collision models: *moving* (false if the collision model is not moving/deforming during the simulation) and *simulated* (false for not moving objects or objects that don't use contact forces, in our case, the US probe).

##### LM method
Collision response with LM method is handled by applying a constraint to colliding objects that prevents interpenetration. In this case, contacts are solved exactly (this method can be interpreted as the penalty method, with infinite *contactStiffness*). To do so, LM method implementation in SOFA is based on some specific components:
1. *FreeMotionAnimationLoop*: this animation loop is required for the LM method. It tells the simulation to initially compute the free motion of all the degrees of freedom (DOFs) without caring about the contacts.
2. *GenericConstraintSolver*: implements projective Gauss Seidel algorithm, an iterative method to find the constraint forces to apply to contacting DOFs to prevent interpenetration, according to Signorini's law.
3. *LinearSolverConstraintCorrection*: required to apply the computed response to the objects of interest.

Keep in mind that scenes relying on the LM method require a direct solver. This means that the default SOFA solver *CGLinearSolver*, which is iterative, will not work.

### Prescribed displacement method
Through this method, nodes on the object surface which are in contact with the US probe are directly displaced. This method does not rely on any collision detection and response component. However, it assumes that the indices of the nodes belonging to the surface mesh which will be moving during the simulation are a-priori known. In our case, we selected the nodes on the breast surface which lie closest to the probe in the initial configuration, and we saved their indices in inData/ground_truth/springnodes_500.txt.

[NOT IMPLEMENTED yet with SofaPython3]


## References
Tagliabue, E., Dall’Alba, D., Magnabosco, E. et al. ["Biomechanical modelling of probe to tissue interaction during ultrasound scanning"](https://link.springer.com/article/10.1007/s11548-020-02183-2) Int J CARS (2020). 

Altair Robotics Lab - University of Verona

Contact: eleonora[dot]tagliabue[at]univr[dot]it
