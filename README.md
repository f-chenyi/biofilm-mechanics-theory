# biofilm-mechanics-theory

Details about about packages installed in the virtual environment can be found in the folder "fenics-envs".

The usage of "interpolate" may be deprecated in versions later than fenics 2017.2.0. To get around this issue, you could assign initial values to the mixed element variable u by directly accessing its dofs. For example, one could import chain in the beginning:

    from itertools import chain

Then get dofs corresponding to a particular scalar element, for example ur, using:

    ur_dofs = np.array(list(chain(ME.sub(0).dofmap().collapse(mesh)[1].values())))
    
Finally the initial values can be given to ur by:

    u.vector()[ur_dofs] = 0.0
    


If you have further questions, please contact Chenyi:cfei@princeton.edu
