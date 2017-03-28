\version "2.10.33"

\header {
  title = "361 - Du Lebensfürst, Herr Jesu Christ"
  composer = "J. S. Bach"
}

global =  {
  \time 4/4 
  \key g \major
}

soprano = \relative c'' {
  \repeat volta 2 {
  \partial 4 g4 
    g a b8 cis d4 
    d cis d b 
    c b a b 
    a2 g4 } a 
  a b c c 
  b8 a b4 a b 
  b b c8 d e4 
  e dis e fis 
  g b, c b 
  a2 a4 d 
  e d c b8 c 
  a2 g4 
}


alto = \relative c' {
  \repeat volta 2 {
  \partial 4 d4 
    e fis d8 e fis4 
    e8 fis g4 fis fis 
    e d8 e fis4 g 
    g fis d } fis8 g 
  a4. gis8 a gis a4 
  a gis e e8 dis 
  e4 e e8 fis g fis 
  fis g a4 g a 
  g8 a b4 b8 a a g 
  g2 fis4 g8 fis 
  e4 fis g8 a d,4 
  e d d4 
}


tenor = \relative c' {
  \repeat volta 2 {
  \partial 4 b4 
    b8 c4 d8 b a a4 
    a a a b8 a 
    g a b c d4 d 
    d4. c8 b4 } d8 e 
  fis e d4 e fis 
  b, e8 d c4 g8 a 
  b c d e c4 cis 
  b b b d4 
  d4 g g,8 a b4 
  e8 d e cis d4 d 
  d8 c c b b a g4 
  g8 fis16 e16 fis4 b4 
}


baixo = \relative c' {
  \repeat volta 2 {
  \partial 4 g8 fis 
    e4 d g fis8 g 
    a4 a, d dis 
    e8 fis g c, c' b a g 
    d'4 d, g } d 
  d'8 c b4 a dis, 
  e e a, e'8 fis 
  g4 gis a ais 
  b b, e d'8 c 
  b a g fis e fis g e 
  cis b cis a d4 b 
  c d e8 fis g4 
  c, d g,4 
}




\score {
  <<
    \new StaffGroup <<
      \override StaffGroup.SystemStartBracket #'style = #'line 
      \new Staff {
        <<
          \global
          \new Voice = "soprano" { \voiceOne \soprano }
          \new Voice = "alto" { \voiceTwo \alto }
        >>
      }
      \new Staff {
        <<
          \global
          \clef "bass"
          \new Voice = "tenor" {\voiceOne \tenor }
          \new Voice = "baixo" { \voiceTwo \baixo \bar "|."}
        >>
      }
    >>
  >>
  \layout {}
  \midi {}
}