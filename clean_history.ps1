# PowerShell script to clean git history of sensitive tokens

# Make sure we're in the right directory
Set-Location -Path "D:\Quickcap v9\App"

# Check if git is initialized
if (-not (Test-Path -Path ".git")) {
    Write-Host "Initializing git repository..."
    git init
}

# Create a backup branch just in case
git branch -m master backup_master

# Create a new orphan branch (this will be our clean branch)
git checkout --orphan clean_master

# Add all files to the new branch
git add .

# Commit the files
git commit -m "Initial commit with clean history"

# Make this the master branch
git branch -D master
git branch -m master

# Now you can push to GitHub
Write-Host "History has been cleaned. You can now push with:"
Write-Host "git push -f origin master"