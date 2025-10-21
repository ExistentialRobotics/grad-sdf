var currentSceneList = ["lego", "ship"];
var currentScene = "lego";
var currentSceneId = 0;

var currentEpoch = "500";
var currentEpochId = 0;
const sceneButtonTextsVideo = {
  0: ["500", "1500", "3000"], // Lego
  1: ["1000", "2000", "5000"], // Ship
};

var currentEpochImage = "500";
var currentEpochIdImage = 0;
const sceneButtonTextsImage = {
  0: ["500", "1000", "1500", "2000", "2500", "3000"], // Lego
  1: ["500", "1000", "1500", "2500", "3500", "5000"], // Ship
};

function ChangeScene(idx){
  var li_list = document.getElementById("scene-result-view-ul").children;
  for(i = 0; i < li_list.length; i++){
      li_list[i].className = "";
  }
  li_list[idx].className = "active";

  currentScene = currentSceneList[idx];
  currentSceneId = idx;

  var li_list = document.getElementById("video-result-view-ul").children;
  for(i = 0; i < li_list.length; i++){
      li_list[i].className = "";
  }
  li_list[0].className = "active";
  currentEpoch = sceneButtonTextsVideo[currentSceneId][0];
  currentEpochId = 0

  // change video paths
  let video = document.getElementById("video_scene")
  old_src = video.src;
  new_scr = old_src.split('/');
  new_scr[new_scr.length-3] = currentScene;
  new_scr[new_scr.length-1] = currentEpoch + ".mp4";
  new_video_dir = new_scr.join('/');
  video.src = new_video_dir;
  // change text for the epoch buttons for the image
  // Get the target button texts for the selected scene
  const newTextsVideo = sceneButtonTextsVideo[currentSceneId] || [];
  // Update the buttons with new text
  const buttonsVideo = document.querySelectorAll("#video-result-view-ul li a");
  buttonsVideo.forEach((buttonVideo, index) => {
      if (newTextsVideo[index] !== undefined) {
          buttonVideo.textContent = newTextsVideo[index];
      }
  });

  var li_list = document.getElementById("image-result-view-ul").children;
  for(i = 0; i < li_list.length; i++){
      li_list[i].className = "";
  }
  li_list[0].className = "active";
  currentEpochImage= sceneButtonTextsImage[currentSceneId][0];
  currentEpochIdImage = 0;

  // change image paths
  let image1 = document.getElementById("scene-image-neuralangelo")
  old_src1 = image1.src;
  new_scr1 = old_src1.split('/');
  new_scr1[new_scr1.length-4] = currentScene;
  new_scr1[new_scr1.length-1] = currentEpochImage + ".png";
  new_image_dir1 = new_scr1.join('/');
  image1.src = new_image_dir1;

  let image2 = document.getElementById("scene-image-ours")
  old_src2 = image2.src;
  new_scr2 = old_src2.split('/');
  new_scr2[new_scr2.length-4] = currentScene;
  new_scr2[new_scr2.length-1] = currentEpochImage + ".png";
  new_image_dir2 = new_scr2.join('/');
  image2.src = new_image_dir2;

  // change text for the epoch buttons for the image
  // Get the target button texts for the selected scene
  const newTextsImage = sceneButtonTextsImage[currentSceneId] || [];
  // Update the buttons with new text
  const buttonsImage = document.querySelectorAll("#image-result-view-ul li a");
  buttonsImage.forEach((buttonImage, index) => {
      if (newTextsImage[index] !== undefined) {
          buttonImage.textContent = newTextsImage[index];
      }
  });

  let mesh1 = document.getElementById("mesh-neuralangelo")
  old_src1 = mesh1.src;
  new_src1 = old_src1.split('/');
  new_src1[new_src1.length-3] = currentScene;
  new_mesh_dir1 = new_src1.join('/');
  mesh1.src = new_mesh_dir1;

  let mesh2 = document.getElementById("mesh-ours")
  old_src2 = mesh2.src;
  new_src2 = old_src2.split('/');
  new_src2[new_src2.length-3] = currentScene;
  new_mesh_dir2 = new_src2.join('/');
  mesh2.src = new_mesh_dir2;

}
// adjusted from ChangeResult
function ChangeEpoch(idx){
  var li_list = document.getElementById("video-result-view-ul").children;
  for(i = 0; i < li_list.length; i++){
      li_list[i].className = "";
  }
  li_list[idx].className = "active";

  currentEpoch= sceneButtonTextsVideo[currentSceneId][idx];
  currentEpochId = idx;
  let video = document.getElementById("video_scene")
  old_src = video.src;
  new_scr = old_src.split('/');
  new_scr[new_scr.length-1] = currentEpoch + ".mp4";
  new_video_dir = new_scr.join('/');
  video.src = new_video_dir;
}

function ChangeEpochImage(idx){
  var li_list = document.getElementById("image-result-view-ul").children;
  for(i = 0; i < li_list.length; i++){
      li_list[i].className = "";
  }
  li_list[idx].className = "active";

  currentEpochImage= sceneButtonTextsImage[currentSceneId][idx];
  currentEpochIdImage = idx;
  let image1 = document.getElementById("scene-image-neuralangelo")
  old_src1 = image1.src;
  new_scr1 = old_src1.split('/');
  new_scr1[new_scr1.length-1] = currentEpochImage + ".png";
  new_image_dir1 = new_scr1.join('/');
  image1.src = new_image_dir1;

  let image2 = document.getElementById("scene-image-ours")
  old_src2 = image2.src;
  new_scr2 = old_src2.split('/');
  new_scr2[new_scr2.length-1] = currentEpochImage + ".png";
  new_image_dir2 = new_scr2.join('/');
  image2.src = new_image_dir2;
}