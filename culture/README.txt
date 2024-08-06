
[This file may describe an older version than the current code]

Trying to make GPTs build their own "culture".

Francois Fleuret
Jun 21st, 2024

* Motivation

The original motivation of this experiment is the hypothesis that
high-level cognition emerges from the competition among humans in the
space of language and ideas.

More precisely, communicating agents try to out-do competitors by
creating stuff that is smart but doable, e.g. some other agents get
it, but not all. Then, that smart thing is added to the "culture",
they all learn and get to understand it, and it repeats.

* Setup

It starts with a "world model" that they got before they communicate,
and from there, they try to "be smart" by proposing quizzes that can
be solved but not by everybody.

There are 5 competing GPTs.

The "world" is a 6x8 grid with three "birds" moving in a straight line
and bouncing on the world's borders. It could be another "world", but
this one has objectness and motion. There are ten colors and 4
directions of motions, so roughly (6x8x4x10)**3 ~ 7e9 states.

Given a random world state, and the state after two iterations of
birds moving, a "quiz" is to predict the second frame, given the
first, or the opposite. The starting and ending states are chosen, by
rejection, so that there is no occlusion.

My home-baked GPT-37M trained with 250k solves this with ~99% success
[to be verified with the new setup].

At every iteration, we select the GPT with the lowest test accuracy,
and run one epoch.

* Creating new quizzes

If its test accuracy got higher than 97.5%, it will create new
quizzes. To do so, it generates a large number of pairs of frames, and
checks which ones of these quizzes are hard but not too hard, which
means [THIS IS THE IMPORTANT BIT]:

  it can be solved, in both time directions, by all the other GPTs
  **but one**

The both time directions is to avoid a simple type of quizzes which is
simply to deal with noise in the first frame.

The GPT generates 1000 of such quizzes, that are added to the
"culture", i.e. the training set.

We update the test accuracy of all the GPTs, and then we go to the
next iteration.

The hope is that interesting concepts emerge (connectivity, symmetry,
interior/exterior, shape vocabulary, etc.)
