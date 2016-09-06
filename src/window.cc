#include "window.h"

#include <SDL.h>
#include <iostream>
#include <stdlib.h>

using namespace rscg;

Window::Window(unsigned w, unsigned h)
{
  if(SDL_Init(SDL_INIT_EVERYTHING) != 0)
    throw std::runtime_error("unable to init SDL2");


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

  _window = SDL_CreateWindow("RealSenseCG",
                                        SDL_WINDOWPOS_CENTERED,
                                        SDL_WINDOWPOS_CENTERED, w, h,
                                        SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);
  _context = SDL_GL_CreateContext(_window);
}

Window::~Window()
{
  SDL_DestroyWindow(_window);
  SDL_GL_DeleteContext(_context);
}

Window::operator::SDL_Window*() const
{
  return _window;
}