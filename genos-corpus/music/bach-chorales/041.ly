\version "2.10.33"

\header {
  title = "41 - Was mein Gott will, das"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \minor
}


soprano = \relative c' {
  \repeat volta 2 {
    \partial 4 e4 
    g4 a4 g4 c4 
    c4 b4 c4 c4 
    b4 a4 d4. c8 
    b4 c4 b2 
    a2. } a4   %5
  a4 a4 b4. a8 
  b4 g4 g4 fis4 
  g4 g4 a4 b4 
  c4 b4 a2 
  b4 e,8 fis8 g4 a4   %10
  g4 c4 c4 b4 
  c4 c4 b4 a4 
  d4. c8 b4 c4 
  b2 a4
  
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 c4 
    e8 d8 c8 d8 e8 f8 g4 
    g4 g4 g4 g4 
    g4 f8 g8 a2 
    gis4 a2 gis4 
    e2. } e4   %5
  d8 e8 fis4 g4. a8 
  g4 e4 e4 d4 
  d4 e4 fis4 f4 
  e8 fis8 gis4 a4. dis,8 
  e4 b4 e4 fis4   %10
  e4 e4 a4 g4 
  g4 a4 a8 g8 g8 f8 
  f4. e8 d4 c8 d8 
  e2 e4 
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 a4 
    b4 a8 b8 c4 c8 d8 
    e8 d16 c16 d4 e4 e4 
    d4 d4 a4 e'4~ 
    e4 e4 e4. d8 
    c2. } c8 b8   %5
  a4 d4 d4. dis8 
  e4 b4 a4 a4 
  b4 c4 a4 d4 
  c8 d8 e4 e8 d8 c8 fis,8 
  gis4 g8 a8 b4 c4   %10
  b4 c4 d4 d4 
  e4 e4 f8 e8 e8 d8 
  d8 c8 b8 a8 e'2~ 
  e8 a,8 gis4 cis
  
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 a4 
    e'4 f4 c8 d8 e8 f8 
    g4 g,4 c4 c4 
    g'4 d8 e8 f8 g8 a4 
    e8 d8 c8 a8 e'2 
    a,2. } a'8 g8   %5
  fis4 e8 d8 g8 d8 g8 fis8 
  e8 b8 e8 d8 c8 a8 d4 
  g,4 c4 c'8 b8 a8 gis8 
  a4 e8 d8 c8 b8 a4 
  e'4 e4 e4 dis4   %10
  e4 a8 g8 fis4 g4 
  c,4 a4 d8 e8 f4 
  b,8 c8 d4 gis,8 e'8 c8 a8 
  e'2 a,4
  
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