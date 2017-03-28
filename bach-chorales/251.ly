\version "2.10.33"

\header {
  title = "251 - Ich bin ja, Herr, in deiner Macht"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key g \minor
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 g4 
    a4 a8 bes16 c16 bes8 g8 d'4~ 
    d8 c16 bes16 a8 g8 fis4 a4 
    bes4 c8 d16 ees16 d8 bes8 f'4~ 
    f8 e16 d16 c8 bes8 a4 d4 
    cis4 d4 e8 f16 g16 f8 e8     %5
    e2 d4 } d4 
  bes8 a8 b8 d8 g8 ees8 c4 
  d8 f8 ees8 d8 ees4 a,4 
  d4 g,4 c8 d16 ees16 d8 c8 
  bes4 a8 g8 g4  %10
  
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 d4 
    ees4 d4 d4. d8 
    g4 fis8 g8 d4 fis4 
    g4 f4 f4. f8 
    bes4 c8 e,8 f4 f4 
    g4 f4 e4 d4     %5
    e8 f8 g4 fis4 } a4 
  g4 g4 g4. ees8 
  aes4 g4 g4 fis4 
  g8 f8 e4 fis8 g8 a4 
  a8 g8 fis4 d   %10
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 bes4 
    c8 bes8 a4 g4. a8 
    bes4 c8 bes8 a4 d4 
    d4 c4 bes4. c8 
    d4 g,8 c8 c4 bes4 
    bes4 a4 a4 a8 d8     %5
    d8 cis16 b16 cis4 a4 } d4 
  d4 d4 c4. c8 
  c4 b4 c4 d4 
  d4 c4 c8 bes8 a8 d8 
  d4 c8 bes8 bes4   %10
  
}

baixo = \relative c' {
  \repeat volta 2 {
    \partial 4 g4 
    g4 fis4 g4. f8 
    ees8 d8 ees8 c8 d4 d4 
    g4 a4 bes4. a8 
    g8 f8 e8 c8 f4 bes4 
    e,4 f4 cis4 d4     %5
    a'4 a,4 d4 } fis4 
  g4 f4 ees4. aes8 
  f8 d8 g4 c,4 c'4 
  bes8 a8 bes8 c8 a8 g8 fis4 
  g d g,   %10
  
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