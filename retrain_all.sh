#!/bin/bash
# Retrain all models, 3 at a time in parallel, output to terminal.
# Run from the PlanarQuadToy directory:
#   bash retrain_all.sh

echo ">>> Batch 1/3: anchored/d6, free/d6, free3d/d6/wp10.0"
python3 train.py --mode anchored --latent_dim 6 --num_steps 50000 --sigma 0.5 --lam 1.0 --w_anchor 10.0 &
python3 train.py --mode free     --latent_dim 6 --num_steps 50000 --sigma 0.5 --lam 1.0 --w_anchor 10.0 &
python3 train.py --mode free3d   --latent_dim 6 --num_steps 50000 --sigma 0.5 --lam 1.0 --w_anchor 10.0 --w_planarity 10.0 &
wait
echo ">>> Batch 1 done"
echo ""

echo ">>> Batch 2/3: free3d/d6/wp1.0, free3d/d6/wp0.2, free3d/d3/wp0.2"
python3 train.py --mode free3d --latent_dim 6 --num_steps 50000 --sigma 0.5 --lam 1.0 --w_anchor 10.0 --w_planarity 1.0 &
python3 train.py --mode free3d --latent_dim 6 --num_steps 50000 --sigma 0.5 --lam 1.0 --w_anchor 10.0 --w_planarity 0.2 &
python3 train.py --mode free3d --latent_dim 3 --num_steps 50000 --sigma 0.5 --lam 1.0 --w_anchor 10.0 --w_planarity 0.2 &
wait
echo ">>> Batch 2 done"
echo ""

echo ">>> Batch 3/3: stiffFree3d — anisotropic stiffness (k_xy vs k_z)"
# k_z/k_xy ratio controls bending: low ratio = bendy, high ratio = stiff
python3 train.py --mode stiffFree3d --latent_dim 6 --num_steps 50000 --sigma 0.5 --lam 1.0 --w_anchor 10.0 --k_xy 1.0 --k_z 0.1 &
python3 train.py --mode stiffFree3d --latent_dim 6 --num_steps 50000 --sigma 0.5 --lam 1.0 --w_anchor 10.0 --k_xy 1.0 --k_z 1.0 &
python3 train.py --mode stiffFree3d --latent_dim 6 --num_steps 50000 --sigma 0.5 --lam 1.0 --w_anchor 10.0 --k_xy 1.0 --k_z 5.0 &
wait
echo ">>> Batch 3 done"
echo ""
echo "All 9 models retrained!"
