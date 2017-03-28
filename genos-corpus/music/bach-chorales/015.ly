\version "2.10.33"

\header {
  title = "15 - Christ lag in Todesbanden"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \minor
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 a4 
    gis4 a8 b8 c4 d4 
    c4 b4 a4 a4 
    f4 g4 a4 g8 f8 
    \time 3/4 e8 d8 e4 d4 }
  \time 4/4 d8 e8 f4 g4 d8 e8 %5
  f4 g4 a4 a4
  d4 cis4 d4 e8 d8
  c4 b4 a4 b4
  c4 a8 bes8 c4 g4
  f4 e4 d2 %10
  a'4 g4 f4 e4
  d1
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 f4 
    f8 e4 gis8 a4 b4~ 
    b8 a4 gis8 e4 e4~ 
    e8 d8 cis8 d8 e4. d8 
    \time 3/4 cis8 d4 cis8 a4 }
  \time 4/4 a4 d4 d8 c8 bes8 c8 %5
  d8 a8 d8 c8 c4 f8 g8
  a8 bes4 a8 a4 e4~
  e8 f4 e8 e4 f4~
  f8 e8 f4 f4 e4~
  e8 d4 cis8 d2 %10
  d4. e8 cis8 d4 cis8
  a1 
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 d8  c8 
    b4 c8 d8 e4 f8 e8 
    e8 f4 e16 d16 cis4 a4 
    a4 bes4 a4 a4 
    \time 3/4 a8 f8 g8 e8 f4 }
  \time 4/4 f8 g8 a4 g8 a8 bes4 %5
  a8 f'4 e8 f4 c4
  f,4 g4 d'8 c8 b4~
  b8 a4 gis8 c4 d4
  g,8 c8 c4 c8 d8 e8 d8
  cis8 a8 bes8 a16 g16 f2 %10
  a8 c8 bes4 a4 g4~ 
  g4 fis8 e8 fis2 
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 d4 
    d4 c8 b8 a8 a'4 gis8 
    a8 f8 d8 e8 a,4 cis4 
    d8 f8 e8 d8 cis8 a8 d4 
    \time 3/4 a'4 a,4 d4 }
  \time 4/4 d4. c8 bes8 a8 g4
  d'8 c8 bes8 c8 f,4 f'8 e8
  d4 e4 fis4 gis4
  a8 f8 d8 e8 a,4 d4
  c4 f,8 g8 a8 bes8 c8 bes8
  a8 f8 g8 a8 bes2 %10
  fis4 g4 a2
  d,1 
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