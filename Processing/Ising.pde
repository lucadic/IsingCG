ising s0;
ising sb1, sb2, sb3, sb4, sb5, sb6, sb10;
float beta=0.44;
float  tmax=0.824;
//Reset button parameter
int set_x = 430, set_y = 640, set_W = 110, set_H = 35;
boolean set_over = false;
HScrollbar hs1, hs2;  
PFont fontA;
void setup() {
 //fullScreen();
  s0 = new ising(120);
  size(600, 700);
  hs1 = new HScrollbar(20, 660, 250, 10, 10);
  background(255);
  
  fontA = loadFont("ArialMT-20.vlw");

  smooth();
}


void draw() {
  int i;
  background(255);
  textFont(fontA, 17);
  fill(0);
 
  sb1 = s0.block(2);
  sb2 = s0.block(3);
  sb3 = s0.block(4);
  sb4 = s0.block(5);
  sb6 = s0.block(6);
  sb10 = s0.block(10);

  beta = 1*(hs1.spos-hs1.sposMin)/hs1.swidth*tmax;
  println(beta);
  fill(0);
  text("Monte Carlo simulation", 55, 40);
  s0.render(30, 60, 2);


  fill(0);
  text("b = 2 ", 430, 40);
  sb1.render(320, 60, 4);


  fill(0);
  text("b = 3", 120, 350);
  sb2.render(30, 370, 6);


  fill(0);
  text("b = 5", 430, 350);
  sb4.render(320, 370, 10);
  fill(0);
  for (i = 0; i<5000; i++) {
    s0.step(beta);
  }

  hs1.update();
  hs1.display();
  fill(255-abs(beta-0.44)*1000, 0, 0);
  textSize(25);
  text("K1 = "+str(float(round((beta-beta%0.001)*1000))/1000), 290, 669);
    
  draw_button(set_x,set_y, set_W,set_H,"reset");
  
  
}




//SCROLL BAR CLASS

class HScrollbar {
  int swidth, sheight;    // width and height of bar
  float xpos, ypos;       // x and y position of bar
  float spos, newspos;    // x position of slider
  float sposMin, sposMax; // max and min values of slider
  int loose;              // how loose/heavy
  boolean over;           // is the mouse over the slider?
  boolean locked;
  float ratio;

  HScrollbar (float xp, float yp, int sw, int sh, int l) {
    swidth = sw;
    sheight = sh;
    int widthtoheight = sw - sh;
    ratio = (float)sw / (float)widthtoheight;
    xpos = xp;
    ypos = yp-sheight/2;
    spos = xpos + swidth/2 - sheight/2;
    newspos = spos;
    sposMin = xpos;
    sposMax = xpos + swidth - sheight;
    loose = l;
  }

  void update() {
    if (overEvent()) {
      over = true;
    } else {
      over = false;
    }
    if (mousePressed && over) {
      locked = true;
    }
    if (!mousePressed) {
      locked = false;
    }
    if (locked) {
      newspos = constrain(mouseX-sheight/2, sposMin, sposMax);
    }
    if (abs(newspos - spos) > 1) {
      spos = spos + (newspos-spos)/loose;
    }
    
    if ( overRect(set_x, set_y, set_W, set_H )) {
    set_over = true;
    }
    else{
      set_over = false;
    }
    
  }

  float constrain(float val, float minv, float maxv) {
    return min(max(val, minv), maxv);
  }

  boolean overEvent() {
    if (mouseX > xpos && mouseX < xpos+swidth &&
      mouseY > ypos && mouseY < ypos+sheight) {
      return true;
    } else {
      return false;
    }
  }


 void display() {
    noStroke();
    fill(204);
    rect(xpos, ypos, swidth, sheight);
    if (over || locked) {
      fill(0, 0, 0);
    } else {
      fill(102, 102, 102);
    }
    rect(spos, ypos, sheight, sheight);
  }

  float getPos() {
    // Convert spos to be values between
    // 0 and the total width of the scrollbar
    return spos * ratio;
  }
}


// Button 

void draw_button(int x, int y, int W, int H, String text){
  fill(255);
  stroke(0);
  rect(x,y, W, H); 
  fill(0);
  text(text, x+W/4, y+3*H/4); 
}
void mousePressed(){
  int i,j;
  if (set_over) {
    for(i = 0; i< s0.d; i++){
      for(j = 0; j< s0.d; j++){
        s0.s[i+ s0.d*j] = -1;
      }
    }
  }
} 

//AUXILIARY FUNCTION FOR SLIDER 
boolean overRect(int x, int y, int width, int height)  {
  if (mouseX >= x && mouseX <= x+width && 
      mouseY >= y && mouseY <= y+height) {
    return true;
  } else {
    return false;
  }
}
// ISING CLASS
class ising {
  int [] s;
  int d;
  ising(int dim) {
    s = new int[dim*dim];
    int i, j;
    d = dim;
    for (i = 0; i<dim; i++) {
      for (j = 0; j<dim; j++) {
        //s[i+dim*j]= 2*int(random(2))-1;
          s[i+dim*j]= -1;
      }
    }
  }
  void step(float beta) {
    int x, y;
    int es = 0;
    float p;
    x = int(random(d));
    y = int(random(d));
    es = 0;
    es = es+s[((x+1)%d) +y*d];
    es = es+s[(x+d-1)%d +y*d];
    es = es+ s[x+d*((y+1)%d)];
    es = es+ s[x+ ((y-1+d)%d)*d];

    p = random(1);
    if (p<=exp(-2 *s[x+d*y]*es*beta)) {

      s[x+d*y]=-1*s[x+d*y];
    } else {
    }
  }

  void render(int x0, int y0, int size) {
    int i, j;
    noFill();

    stroke(75,0,130);
  
    for (i = 0; i<d; i++) {
      for (j = 0; j<d; j++) {
        if (s[i+d*j] ==1) {

          fill(255,215,0);
          rect(x0+size*i, y0+size*j, size, size);
          
        } else {
          fill (75,0,130);

          rect(x0+size*i, y0+size*j, size, size);
        }
      }
    }
  }
  ising block(int size) {
    int i, j, ir, ic;
    int stemp;
    ising out;
    out = new ising(d/size);
    if (d%2 ==0) {
      for (i = 0; i<d/size; i++) {
        for (j=0; j<d/size; j++) {
          stemp = 0;
          for (ir = 0; ir<size; ir++) {
            for (ic = 0; ic <size; ic++) {
              stemp +=s[size*i+size*j*d+ir+ic*d];
            }
          }
          if (stemp >0)
            out.s[i+d*j/size] = 1;
          if (stemp <0)
            out.s[i+d*j/size] = -1;
          if (stemp==0) {
            if (random(1)<0.5) {
              out.s[i+d*j/size] = 1;
            } else {
              out.s[i+d*j/size] = -1;
            }
          }
        }
      }
    }
    return out;
  }
}
