#include <iostream>

#include <SDL2/SDL.h>

#include <glbinding/gl/gl.h>
#include <glbinding/Binding.h>

#include <librealsense/rs.hpp>

#include "realsenseimage.h"

int
main(int argc, char **argv)
{
  using namespace gl;

  if(SDL_Init(SDL_INIT_EVERYTHING) != 0)
  {
    std::cerr << "unable to init SDL2" << std::endl;
    return -1;
  }

  SDL_GL_SetAttribute(SDL_GL_RED_SIZE,    8);
  SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE,  8);
  SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE,   8);
  SDL_GL_SetAttribute(SDL_GL_ALPHA_SIZE,  8);
  SDL_GL_SetAttribute(SDL_GL_BUFFER_SIZE, 32);

  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

  SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 16);
  SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);

  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);

  SDL_Window *window = SDL_CreateWindow("RealSenseCG",
                                        SDL_WINDOWPOS_CENTERED,
                                        SDL_WINDOWPOS_CENTERED, 800, 600,
                                        SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);
  SDL_GLContext context = SDL_GL_CreateContext(window);


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
  while(running)
  {
    while(SDL_PollEvent(&event))
    {
      if((event.type == SDL_QUIT) or ((event.type == SDL_KEYUP) and
                                     event.key.keysym.sym == SDLK_ESCAPE))
        running = false;
    }
    device->wait_for_frames();

    depthImage.update(*device);

    glClearColor(1.f, 1.f, 1.f, 1.f);
    glClear(gl::GL_COLOR_BUFFER_BIT);

    SDL_GL_SwapWindow(window);
  }

  SDL_GL_DeleteContext(context);
  return 0;
}
