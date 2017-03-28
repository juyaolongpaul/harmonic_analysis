\version "2.10.33"

\header {
  title = "13 - Allein zu dir, Herr Jesu Christ"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key c \major
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 c4 
    g4 a8 b8 c8 d8 e4 
    d4 c2 b4 
    c2\fermata r4 e4 
    d4 c4 b8 a8 g4 
    a4 b8 c8 d4 c8 b8     %5
    c4 b4 a4\fermata } e'4 
  f8 e8 d4 e8 d8 c4 
  d4 e4 a,4\fermata e'4 
  f8 e8 d4 e8 d8 c4 
  d4 e4 a,4\fermata c4   %10
  b4 a4 g4 a4 
  f8 e8 f4 e4\fermata a4 
  g4 c2 b4 
  c4\fermata g4 a4 b4 
  c8 d8 e4 d4 c4   %15
  b4 b4 a4\fermata
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 e4 
    e8 d8 c8 d8 e8 f8 g4 
    g4. fis8 g2 
    g2 r4 g4 
    g8 f8 e4 e e
    e8 fis8 gis8 a8 b4 a4     %5
    a4 gis4 e4 } a4 
  a4 a4 gis4 e4 
  a4 g4 f4 e4 
  a4 a8 g8 g4 g4 
  a4 b4 e,4 g4   %10
  g8 fis8 e4 e8 g8 f8 e8 
  d8 cis8 d4 cis4 c4 
  d4 g8 f8 e4 d4 
  e4 e4 fis4 gis4 
  a4 e4 a2~   %15
  a4 gis4 e4 
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 a8  b8 
    c8 b8 a4 g4 g8 a8 
    b8 a16 b16 c8 d8 e4 d4 
    e2 r4 c4 
    b4. a8 g8 a8 b4 
    c4 d8 e8 f8 e8 e4     %5
    f8 d8 b8 c16 d16 c4 } c4 
  c4 b8 a8 b4 c8 b8 
  a4 a4 d4 b4 
  c4 d4 c4 c4 
  c4 b4 c4 e4  %10
  d4 e4 b4 cis4 
  a4 a4 a4 a4 
  b8 a8 g2 g4 
  g4 e'4 d4 d 
  e8 d8 c8 b8 a8 b8 c8 d16 e16   %15
  f4 e8 d8 cis4 
}

baixo = \relative c' {
  \repeat volta 2 {
    \partial 4 a4 
    e4 f4 e8 d8 c4 
    g'4 a4 g4 g4 
    c,2\fermata r4 c4 
    g'4 a4 e4. d8 
    c8 c'8 b8 a8 gis4 a4     %5
    d,4 e4 a,4\fermata } a'4 
  d,8 e8 f4 e4 a8 g8 
  f8 e8 d8 cis8 d4\fermata gis4 
  a4 b4 c4 e,4 
  fis4 gis4 a4\fermata e8 fis8   %10
  g4 cis,8 dis8 e4 a,4 
  d8 f8 e8 d8 a'4\fermata f4~ 
  f4 e8 d8 e8 f8 g4 
  c,4\fermata c'4~ c8 d8 c8 b8 
  a8 b8 a8 g8 f4. e8   %15
  d4 e4 a,4\fermata
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