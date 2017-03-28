\version "2.10.33"

\header {
  title = "3 - Ach Gott, vom Himmel sieh' darein"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key e \minor
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 b4 
    c4 b4 a4 e'4 
    e8 d8 c4 b4 d4 
    c4 b4 a4 b8 c16 d16 
    c4 b4 a4 } a4 
  b8 a8 g4 fis4 e8 fis8   %5
  g4 a4 b4 g4 
  a8 b8 c4 b4 c8 b8 
  a4 gis4 a4 a4 
  e'4 c4 d4 e4 
  d4 c4 b4 
}

alto = \relative c'' {
  \repeat volta 2 {
    \partial 4 gis4 
    a4 gis4 a4 gis8 a8 
    b4 e,8 fis8 gis4 gis4 
    a4 gis4 a8 g8 f4 
    e2 e4 } d4 
  d8 dis8 e4 dis4 e8 dis8   %5
  e8 g8 fis8 e8 dis4 b4 
  a4 a'4 gis4 a4 
  e4 e4 e4 e4 
  e4 e4 f4 g4 
  fis8 gis8 a4 e4 
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 e4 
    e4 d4 e4 d8 c8 
    b4 c8 d8 e4 f4 
    e4 e8 d8 c4 d4 
    gis,8 a4 gis8 c4 } a4 
  g8 a8 b4 b4 b8 a8   %5
  b4 c4 fis,4 e'8 d8 
  c8 d8 e4 e4 e8 d8 
  c4 b4 c4 c4 
  b4 a4 a4 bes4 
  a8 e8 fis4 gis4 
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 e4 
    a4 b4 c4 b8 a8 
    gis4 a4 e4 b4 
    c8 d8 e4 f4 e8 d8 
    e2 a,4 } fis'4 
  g8 fis8 e4 b'8 a8 g8 fis8   %5
  e8 d8 c4 b4 e4 
  f4 c8 d8 e4 a,8 b8 
  c8 d8 e4 a,4 a'4 
  gis4 a8 g8 f8 e8 d8 cis8 
  d4 dis4 e4 
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