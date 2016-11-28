
## Docker Cloud Node

### Docker Authorized Keys for SSH

Following the instructions at the following link to setup SSH into the Docker Cloud Node:

https://docs.docker.com/docker-cloud/infrastructure/ssh-into-a-node/

### Install git Client

* SSH into the Docker Cloud Node and install the git client:

`apt-get install git`

### Clone emotional-faces repo

`git clone https://github.com/dwdii/emotional-faces.git`


### Clone facial_expressions repo

`git clone https://github.com/dwdii/facial_expressions.git`
	
### Swap File

https://www.centos.org/docs/5/html/5.2/Deployment_Guide/s2-swap-creating-file.html

https://forums.docker.com/t/docker-swap-space/3908/3

```
dd if=/dev/zero of=/var/lib/docker/swapfile bs=1024 count=16777216

mkswap /var/lib/docker/swapfile

swapon /var/lib/docker/swapfile

vi /etc/fstab 
```

Add the following line to the fstab file:

`/swapfile swap swap defaults 0 0`




