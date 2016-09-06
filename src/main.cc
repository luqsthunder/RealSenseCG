#include <iostream>

#include <SDL2/SDL.h>

#include <glbinding/gl/gl.h>
#include <glbinding/Binding.h>

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include <librealsense/rs.hpp>

#include "realsenseimage.h"
#include "shaderprogram.h"
#include "window.h"

int
main(int argc, char **argv)
{
  using namespace gl;

  rscg::Window window{};

  glbinding::Binding::initialize();

  rs::context ctx;
  if(ctx.get_device_count() == 0)
    throw std::runtime_error("no device found");

  rs::device *device = ctx.get_device(0);
  device->enable_stream(rs::stream::depth, 640, 480, rs::format::z16, 60);
  device->start();

  std::cout << "device scale: " << device->get_depth_scale() << std::endl;

  bool running = true;
  SDL_Event event;
  rscg::RealSenseImage depthImage{640, 480};

  rscg::ShaderProgram shaderProgram{"Shaders/simpleshader",
                                    {GL_VERTEX_SHADER, GL_FRAGMENT_SHADER}};

  glm::mat4 proj;
  //proj = glm::ortho()
  while(running)
  {
    while(SDL_PollEvent(&event))
    {
      if((event.type == SDL_QUIT) or ((event.type == SDL_KEYUP) and
                                     event.key.keysym.sym == SDLK_ESCAPE))
        running = false;
    }
    device->wait_for_frames();

    //depthImage.update(*device);

    glClearColor(1.f, 1.f, 1.f, 1.f);
    glClear(gl::GL_COLOR_BUFFER_BIT);

    //shaderProgram.updateUniform(proj, "");
    depthImage.draw(shaderProgram.programID());

    SDL_GL_SwapWindow(window);
  }

  device->stop();

  return 0;
}
