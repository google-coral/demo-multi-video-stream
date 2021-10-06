/*
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/*
 * Utility.h
 *
 *  Created on: Sep 9, 2021
 *      Author: pnordstrom
 */

#ifndef SRC_UTILITY_H_
#define SRC_UTILITY_H_

namespace szd {
constexpr double EPSILON = 1E-9;

class Utility {
 public:
  struct Point {
    Point()
        :
        x_(0),
        y_(0) {
    }
    Point(float x, float y)
        :
        x_(x),
        y_(y) {
    }
    Point(const Point &p)
        :
        x_(p.x_),
        y_(p.y_) {
    }
    Point(const std::pair<float, float> &p)
        :
        x_(p.first),
        y_(p.second) {
    }
    // Return distance between this point and the point p.
    const double GetDistance(const Point &p) const;
    // Get the direction for this point and point b, c.
    // Return value is an int with value:
    // 0 if the three points is collinear
    // 1 if the direction is clockwise
    // 2 if the direction is counter clockwise
    const int GetDirection(const Point &b, const Point &c) const;
    float x_;
    float y_;
  };

  struct Line {
    Line(const Point &begin, const Point &end)
        :
        begin_(begin),
        end_(end),
        length_(begin.GetDistance(end)) {
    }
    // Checks whether if this line contains the point p.
    bool ContainsPoint(const Point &p) const;
    // Checks whether if this line intersects the line l.
    bool IntersectsLine(const Line &l) const;
    Point begin_, end_;
    double length_;
  };

  class Polygon {
   public:
    Polygon() {
    }
    Polygon(std::vector<Point> &polygon_points);
    // Return a vector of lines in this polygon.
    const std::vector<Line>& GetLines() const {
      return lines_;
    }
    // Return a reference to the string representing this polygon in svg
    // form.
    const std::string& GetSvgStr() const {
      return svg_str_;
    }
    ;
    // Set svg string.
    void SetSvgStr(const std::string &svg) {
      svg_str_ = svg;
    }
    ;
   private:
    std::vector<Line> lines_;
    std::string svg_str_ { "None" };
    int current_width_, current_height_;
  };

  class Box {
   public:
    Box(const float x1, const float y1, const float x2, const float y2);
    // Return true if this box collided with the polygon p.
    const bool CollidedWithPolygon(const Polygon &p,
                                   const float max_width) const;
    // Return true if this box collided with the line l.
    const bool IntersectsLine(const Line &l) const;

   private:
    std::vector<Point> points_;
    std::vector<Line> lines_;
  };

  Utility();
  Utility(const Utility &other) = delete;
  Utility(Utility &&other) = delete;
  Utility& operator=(const Utility &other) = delete;
  Utility& operator=(Utility &&other) = delete;
  virtual ~Utility();
};

} /* namespace szd */

#endif /* SRC_UTILITY_H_ */
