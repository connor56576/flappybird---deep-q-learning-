# flappybird---deep-q-learning-
My first project with deep learning. I made a pygame recreation of flappy bird then used pytorch to train a reinforcement based model. Try it out!

I have included an already somewhat trained model for you to use, however you can always create a completely new model by not adding the dqn_flappy.pth file to the directory or changing the name!

As this was my first project with deep learning, I expect my code to not be perfect, feel free to change any code you want to.

There includes different modes within the code, such as:

human (play yourself) 
        
render (shows each epoch while running, much slower than normal training)
<img width="1587" height="924" alt="image" src="https://github.com/user-attachments/assets/107fc0be-3b96-4ef4-a8ae-c2cbde9f3292" />

        
visulise (create a matplotlib graph while training to track the average and best rewards, and shows the epsilon decay over time)
<img width="1906" height="1029" alt="image" src="https://github.com/user-attachments/assets/5a66aff0-c542-4a3c-beeb-0157f6349094" />        

To train a model, make sure all files are in the same directory and run the agent.py file (I used VS code throughout). 

To test a trained model, run the eval_flappy.py file, make sure that the model path is the model you want to test.

module dependancies include pytorch, pygame, numpy, matplotlib.

Let me know what you think! My model achieved a peak score of 22, and was trained for only a few hours. Try and see if you can do better! 


