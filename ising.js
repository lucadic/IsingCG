//Ising model class
class isingSim{


    constructor(L,beta ){
      //  L : linear size of the system
      //  beta: inverse temperature
      this.p = [];
      this.beta = beta;
      this.N = L*L;
      this.L = L;
      //Initialization of the configuriation
      for(let i = 0;i<this.N;i++){
        this.p.push(1);
      }
    }
  
  
    localField(x,y){
      // compute the local field of the spin in position (x,y) 
      let ix,iy;
      let h =  0;
      ix = x+1 %L;
      iy = y;
      h = h+ this.p[ix+this.L*iy];
      ix = (x-1+L)%L;
      iy = y;
      h = h+ this.p[ix+this.L*iy];
      ix = x;
      iy = (y+1)%L;
      h = h+ this.p[ix+this.L*iy];
      ix = x;
      iy = (y-1+L)%L;
      h = h+ this.p[ix+this.L*iy];
      return h;
    }
  
  
    m(){
      //Compute the magnetization of the system 
      let out = 0;
      for(let i = 0;i<this.N;i++){
        out += this.p[i];
      }
      return out;
    }
  
  
    McStep(){
      //Performs one montecarlo step 
      let x = 0;
      let y = 0;
      let p1,p2;
      x = Math.floor(random(this.L));
      y = Math.floor(random(this.L));
      let h = this.localField(x,y);
      p1 = this.beta*h*this.p[x+L*y];
      if(p1<=0){
        this.p[x+this.L*y]*=-1;
      }
      else{
        p1 = Math.exp(-2*p1);
        p2 = random(0,1);
        if (p2 <p1){
          this.p[x+this.L*y]*=-1;
        }
      }
    }
  
  
    plot(l, x0, y0){
      //plot the ising lattice in postion x0,y0
      //  l: size of the spin block
      //  x0 : x coordiante
      //  y0 : y coordinate
      for(let i = 0;i<this.L;i++){
        for(let j = 0;j<this.L ;j++){
          //let c = (0,0,0);
          //print(this.p[i+L*j]);
          if (this.p[i+L*j] ==1){
            let c = color(0,0,70);
            fill(c);
            noStroke();
            rect(l*i+x0,l*j+y0,l,l);
          }
          if (this.p[i+L*j]==-1){
            let c = color(226, 226, 226);
            fill(c);
            noStroke();
            rect(l*i+x0,l*j+y0,l,l);
          }
        }
      }
      }
  
  
  plot_CG(bs, l, x0, y0){
      //plot the coarse grained ising lattice in postion x0,y0
      //  l: size of the spin block
      //  x0 : x coordiante
      //  y0 : y coordinate
      //  bs : coarse graining block size
      for(let i = 0;i<this.L/bs ;i++){
        for(let j = 0;j<this.L/bs ;j++){
          var site = 0;
          var P = 0;
          for( let kx = 0; kx<bs; kx++){
            for( let ky = 0; ky<bs; ky++){
              site += this.p[i*bs+kx+(j*bs+ky)*this.L]
            }
          }
  
          if (site > 0){
            P = 1;
          }
          if (site < 0 ){
            P = -1;
          }
          if (site == 0){
            if (Math.random()>0.5)
              P = -1;
            else 
              P = 1;
          }
          //let c = (0,0,0);
          //print(this.p[i+L*j]);
  
          if (P ==1){
            let c = color(0,0,70);
            fill(c);
            noStroke();
            rect(l*i+x0,l*j+y0,l,l);
          }
          if (P==-1){
            let c = color(226, 226, 226);
            fill(c);
            noStroke();
            rect(l*i+x0,l*j+y0,l,l);
            }
  
          //circle((i+1)*l,(j+1)*l,l);
  
          }
        } 
      }
  
  
      
    }   