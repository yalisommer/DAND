
Some Notes:

Anchored = interior vertices trainable
Free = all trainable but no z displacement
Free3d = all trainable

Seed config is the flat plane (z=0 in the neural function maps to this)

For training:
Each step, sample batch of random latent vecors from standad dist (default batch size of 32, can use --batch_size to modify)

Decoder maps each vector to a mesh config

Compute loss (physically based) measuring how good each config is. (Low energy deformations)

Backprop + update network weights

Do this 50k iters (--num_steps)

_________________

Decoder is MLP (5 layers, 64 each, ELU activations, taken from paper)

final vertex positions computed as f(z) = seed + MLP(z)
- So the network learns displacement from the base config

(note we have a p factor that increases as step/num_steps, where f(z) = seed + p*MLP(z) to make the deformation space grow outwards as we continue training)

_________________

**LOSS FUNCTION**

Loss = L_energy + lambda·L_metric + w_anchor·L_anchor

*L_energy = average over a batch of E_pot(f(z))*

E_pot varies based on which type we use, but generally it is made up of:

Edge energy: `Σ (‖vᵢ - vⱼ‖ - L₀)²` | Edges that stretch or compress vs. rest lengths
- Restricts stretchinf & compression
- L₀ = rest length of edge in undeformed mesh

Area energy: `Σ (A_quad - A₀)²` | Quads whose area deviates from rest area (1.0)
- Restricts shearing/collapse even if edge lengths are preserved
- A₀ = rest area of face in undeformed mesh

XY displacement: `k_xy · Σ(Δxᵢ² + Δyᵢ²)` | In-plane movement from rest position

Z displacement: `k_z · Σ zᵢ²` | Out-of-plane movement (z-coordinate)

- These two are used in tandem to "ratio" how much a mesh should deform in regards to z or not

*L_metric* = average over pairs of [ log( ‖f(z) - f(z')‖ / (σ_eff · ‖z - z'‖) ) ]²


Overall: decoder should be distance preserving, ie nearby latent values should give similar meshes

(this idea is taken pretty directly from the paper)

note: 
- sigma = how much dist in config space corresponds to dist in latent space
- lambda = weight of metric loss relative to energy loss

(defaulted to .5, and 1, I havent played around much with these)

*L_anchor* = avg (MLP(0)^2)

Here to ensure that the MLP(0) is 0, so that f(0) remains = seed

We can also scale this with a w_anchor param in case it doesnt stick, but I've found that networks are pretty good at keeping the f(0) as the seed.

--SIDENOTE--
for less clear meshes, like box and hemisphere, kxy is stiffness along the surface tangent, kz is along the surface normal
lower coefficients = cheaper motion in that direction

Meshes of Interst:


python3 latent_viz.py --model checkpoints/anchored/d6/model.pt

python3 latent_viz.py --model checkpoints/free/d6/model.pt



python3 latent_viz.py --model checkpoints/stiffFree3d/d6/kxy1.0_kz0.1/model.pt

python3 latent_viz.py --model checkpoints/stiffFree3d/d6/kxy1.0_kz5.0/model.pt



python3 latent_viz.py --model checkpoints/stiffFree3d/box/d6/kxy1.0_kz0.5/model.pt

python3 latent_viz.py --model checkpoints/stiffFree3d/box/d6/kxy1.0_kz5.0/model.pt



python latent_viz.py --model checkpoints/stiffFree3d/hemiTri/d4/kxy1.0_kz0.5/model.pt

//notes:
energy on the diagonals

planarity: distance between diagonals / avg length of the diagonals in percentages (mult by 100)

"If we can sample from the valid space, we could find constraints"

How can we find constraints in a penalty method.

Minimize planarity as a large part of loss.

Take our current loss + a large coefficient (ie 10k more) on planarity
solve the system constraint wise

find references of people trying to learn optimization with constraints
- if anyone had a system of constraints where we conditioned a network within the constraints

should ultimately have some system of constraints, such that they equal 0 and we need to satisfy them. Q is how.
We know how to do shapespace through loss, how do we also optimize for a constraint
So our first test will be loss + huge penalty of constraint squared
We also want to solve for this system "lagrange multiplier"

Adverserial way to learn a constraint?
Randomly sample space, theyre bad in constraint, so move towards the constraint

Maybe within the batch step, take only valid options within this batch, then look at loss for those
Or, bring all batches into the valid constraints
Could be a problem others have approached!!!!
- Look into it!

Others intersted in this project?
- Z disentanglement -> othogonal space

Goals for this week:
Look into other papers where people are trying to bake constraints into a system

Add a penalty system

Vizualize planarity -> color per face. Show colorbar as key for this.

When I touch a vertex, maybe I can do a callback event? Mimick allowing users to modify vertexes directly instead of sliders.

Look into lagrange mutliplier/networks -> lagrange is keyword for constraints

Look into disentanglement, but reach goal, no worries now

New stuff:


New loss implemented

Two versions. 

Main one I did:


the planarity per face is dist/average length + eps (to prevent zero cases)

with dist = |(v1-v0) ⋅ n | / ||n|| + eps

and avg length = ||d1|| + ||d2|| / 2


Initial proposed ver Ver: 

same avg len, but the dist is || of the closest vector between two points. This is solved through linear system to find this closest vector.

This method made it far too rigid and was a bit of a time sink.

Notes:

Look at more sophisticated penalty metrics. Be more exact rather than just a simple planarity lambda.

Could also try other constraints, like length constraints. Some form of piecewise rigidity. Stuff we can just make up.
Inequality constraints too? Can deform an edge but only like 10% more or less stretch or compression. Lets see effectivity.

Read literature on this!!!!!

Train on some larger architectural meshes


Maybe need bigger batch sizes for larger meshes


Concern: Energy should be an average in relation to the number of faces. Right now we may be getting huge energy just bc we have more faces.

Lets also graph average energies as we go. Should graph different energies seperately. 
With a lambda of 500, should expect it to be 500 times more potent. 
Could graph all parts of the loss seperately?


Notes:

for ALM:
idea is no lambda entered for planarity, instead the lambda is learned throughout training as the condition is satisfied

for rho, we set the bounds of where lambda can go to throughout training

for GINN:
restructure training:
feasibility (is the output geometrically valid?), diversity (does the latent space explore meaningfully?), and objective (is the energy low?).

only feasible samples vote on the energy loss. If a decoded mesh violates planarity beyond a threshold ε, it's masked out of the energy gradient entirely. The model is forced to find the feasible region first, then minimize energy within it

What ive done:
found these papers. Tried some rough implementations to pretty bad results
Generally marked by lots of issues... bad lip detection for meshes caused me a lot of problems + lots of issues with mesh loading.
I tried to implement a mesh floor detection algorithm that was a time sink, now we just rotate to fix y-up z-up incompatibilities
- need to streamline floor detection

added energy averaging, but i think this had bad consequences on smaller meshes. need to a deeper dive into this.
- could explain many other failures

added some other penalties, some interesting, but this took a back seat


notes:
need to maybe regularize better?
energies coscaled

need to satisfy the lagrange eq and the constraints being zero
augmented lagrange methods combine penalty 

ginn -> 
issue with planarity, is that its hard to optimize and randomize. totally failed for amir last time.
approach for having things naturally move in the shape space.

amir anti-rejection.

hes in favor of learning the shape space as a lagrangian of functions
such a way that the lagrangian parameteries the shape space
- this can even be a complex function
- lagrange the normals to the space of constraints
- if we have a 1d constraint, the lagrangian is the scalar in the direction to the normal of this constraint
- if constraint complicated, lambda will be complicated
- start more with working penalty method is effective, then move onto methods to learn manifold with proper lagrangian

Goals for the week:
streamline. look into paper one, undstand energies.

finish the penalty method. Focus on moving parameters, getting a nice deformation that does not kill the planarity

rethink smoothness params. Use direclet energy-- the deformation field is a function from point to point. We want smoothness, punish field minus field on an edge. Punish the smoothness of the velocity. Edges are the quad edges & the diagonals. Then, the deformation field will be smooth across the mesh. Don't deal with tangent field stuff. Can look into direclet seperation along axis? but this could just be nice.

Look into the energies amir used in his paper, maybe use these.
read amir's paper?
but dont want to use a subspace, use my space. just look at the energies I used.
fine maps preserve planarity. if we have a deformation using a single fine map/linear map, it will preserve planarity, but will kill interesting transformations.

find a way to write them as a deformation fields?

write to Amir what I understand from his paper about the energies.