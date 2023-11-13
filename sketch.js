

let p = [];
let count = 0;
let noise = 1;
let L = 120;
let N = L*L;
let b = 1;
let simulationRate = 0.1;
let M = 0;
let sim;
let frame_y_slider = 25;
let frame_x = 0, frame_y = 170;
function setup() {
  let w = 0;
  //w = getWidth();
  w = 600;

  var myCanvas = createCanvas(w,800);
  b = int(w/L);
  myCanvas.position(40,0);
  myCanvas.background(255,255,255);
  sim = new isingSim(L,0.3);
  myCanvas.parent('Ising');
  //Create the slider for the coupling constant K 
  slider = createSlider(0, 100, 40);
  slider.position(60,55+frame_y_slider);
  slider.style('width', '250px');
  PlayButton = createButton('Reset');
  // put button in same container as the canvas
  PlayButton.parent('Ising')
  //  button.size(90,30);
  PlayButton.position(480,50+frame_y_slider);
  PlayButton.mouseClicked(reset);
  PlayButton.style("font-size", "20px");
}


function draw() {

  sim.beta = slider.value()/100;
  background(255,255,255,100);

  // Text rendering
  textSize(30); 
  fill(200,30,30);
  text('ISING MODEL AND KADNOFF BLOCKING ', 0, 45);
  textSize(22);
  fill(0,0,0);
  text('Montecarlo Simulation',10,frame_y-20);
  text('block size = 2',10+300+45,frame_y-20);
  text('block size = 3',10+45,frame_y-20+300);
  text('block size = 4',10+300+45,frame_y-20+300);
  textSize(26);
  text("K1 = "+str(round(sim.beta,2)),285,75+frame_y_slider);


  M = 0;
  for(let i = 0;i<simulationRate*sim.N;i++){
    sim.McStep();
}
  sim.plot(2,frame_x,frame_y);
  sim.plot_CG(2,4,300+frame_x,frame_y);
  sim.plot_CG(3,6,frame_x,300+frame_y);
  sim.plot_CG(4,8,300+frame_x,300+frame_y);

}



function reset(){
  // reset the lattice to the initial configuration (all spins up )
  for(i = 0; i<sim.N;i++){
    sim.p[i] = 1;
  }
}




function getWidth() {
  if (self.innerWidth) {
    return self.innerWidth;
  }

  if (document.documentElement && document.documentElement.clientWidth) {
    return document.documentElement.clientWidth;
  }

  if (document.body) {
    return document.body.clientWidth;
  }
}
