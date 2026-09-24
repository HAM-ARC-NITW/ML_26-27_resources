### How to create a Pull Request on this repository:

(this is obviously assuming you all have git installed on your system, and have your gh credentials logged in)

##### Step 1: Fork the repo
select the *fork button* found at the top right of the main page.
this creates your own copy of the repository under your github account. you will make changes to the repo locally on your laptop and then submit them later via a pull request.

##### Step 2: Clone your fork
open a terminal and run:

`
git clone https://github.com/<your-username>/<repo>.git
`

(if youre on windows, open the **git bash** in your desired folder, everything that follows stays the same)

now run this command:

`
git remote add upstream https://github.com/HAM-ARC-NITW/ML_26-27_resources.git
`

verify by running:

`
git remote -v
`

it should show origin as your personal account and upstream as the organisations' url (this helps with pulling changes later)
```
origin   https://github.com/<your-username>/<repo>.git
upstream https://github.com/HAM-ARC-NITW/ML_26-27_resources.git
```

##### Step 3: fetch the latest work before you do anything
```
git fetch upstream
git checkout main
git pull upstream main
```

##### Step 4: create a branch for your task
dont work directly on `main`, make a new branch instead

```git checkout -b <your-topic-name>```

##### Step 5: add your content
if you find a temporary file inside your respective folder, delete it and add your stuff.
please follow the naming conventions and structure as in the previous sessions.
open a vscode window (or an editor of your choice) and add whatever ya got.

also add the relevant indexing in the main [readme](./README.md) file. you can check the syntax inside the .md file to figure out how to do it.

##### Step 6: push the changes to your local branch
first add all the files to track:
``` git add .```
then verify with ```git status```

now commit to your branch:
```git commit -m "yay new contentttt"```

now push the whole thing to your fork
```git push -u origin <your-topic-name>```
(note that the 'your-topic-name' field is the name of the branch you created, so make it sure it matches)

##### Step 7: Create a PR
go to your fork on github and click **compare and pull request**
fill out the necessary stuff there and submit. we'll review it and approve.
